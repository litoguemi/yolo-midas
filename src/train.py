import argparse
import configparser as cp
import logging

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils import test  # import test.py to get mAP after each epoch
from model.mde_net import MDENet
from model.monocular_depth_estimation_dataset import *
from utils.utils import *
from utils.parse_config import *
from torch.utils.tensorboard import SummaryWriter

# Constants
MIXED_PRECISION_URL = 'https://github.com/NVIDIA/apex'
HYP_FILE_PATTERN = 'hyp*.txt'
CONFIG_FILE = "cfg/mde.cfg"
WEIGHTS_DIR = 'weights'
RESULTS_FILE = 'results.txt'
LAST_MODEL = os.path.join(WEIGHTS_DIR, 'last.pt')
BEST_MODEL = os.path.join(WEIGHTS_DIR, 'best.pt')

# Global values
mixed_precision = None
hyp = None
conf = None
yolo_props = None
freeze, alpha = None, None
device = None    
tb_writer = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_mixed_precision():
    try:
        from apex import amp
        return True
    except ImportError:
        logging.warning(f'Apex not installed. Recommended for mixed precision and faster training: {MIXED_PRECISION_URL}')
        return False

def load_hyperparameters():
    hyp = {
        'giou': 3.54, 'cls': 37.4, 'cls_pw': 1.0, 'obj': 64.3, 'obj_pw': 1.0,
        'iou_t': 0.225, 'lr0': 0.01, 'lrf': 0.0005, 'momentum': 0.937, 
        'weight_decay': 0.000484, 'fl_gamma': 0.0, 'hsv_h': 0.0138, 
        'hsv_s': 0.678, 'hsv_v': 0.36, 'degrees': 0.0, 'translate': 0.0, 
        'scale': 0.0, 'shear': 0.0
    }
    hyp_files = glob.glob(HYP_FILE_PATTERN)
    if hyp_files:
        logging.info(f'Using {hyp_files[0]}')
        hyp_values = np.loadtxt(hyp_files[0])
        for key, value in zip(hyp.keys(), hyp_values):
            hyp[key] = value
    return hyp

def log_focal_loss(hyp):
    if hyp['fl_gamma']:
        logging.info(f'Using FocalLoss(gamma={hyp["fl_gamma"]})')

def read_config():
    conf = cp.RawConfigParser()
    conf.read(CONFIG_FILE)
    return conf

def get_yolo_properties(conf):
    return {
        "anchors": np.array([float(x) for x in conf.get("yolo", "anchors").split(',')]).reshape((-1, 2)),
        "num_classes": conf.get("yolo", "classes")
    }

def get_freeze_alpha_values(conf):
    freeze, alpha = {}, {}
    for layer in ["resnet", "midas", "yolo", "planercnn"]:
        freeze[layer], alpha[layer] = (True, 0) if conf.get("freeze", layer) == "True" else (False, 1)
    return freeze, alpha


def extend_img_size(opt):
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # Extend to 3 sizes (min, max, test)


def select_device_and_precision(opt):
    global mixed_precision
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False
    return device


def initialize_tensorboard(opt):
    if not opt.evolve:
        try:
            tb_writer = SummaryWriter()
            print("Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/")
        except:
            tb_writer = None
    else:
        tb_writer = None
    return tb_writer


def download_evolve_file(opt):
    if opt.bucket:
        os.system(f'gsutil cp gs://{opt.bucket}/evolve.txt .')  # Download evolve.txt if exists


def select_parent(x, n, fitness, method='single'):
    w = fitness(x) - fitness(x).min()  # Weights
    if method == 'single' or len(x) == 1:
        return x[random.choices(range(n), weights=w)[0]]  # Weighted selection
    elif method == 'weighted':
        return (x * w.reshape(n, 1)).sum(0) / w.sum()  # Weighted combination


def mutate_hyperparameters(x, hyp, method=3, mp=0.9, s=0.2):
    npr = np.random
    npr.seed(int(time.time()))
    g = np.array([1] * len(hyp))  # Gains
    v = np.ones(len(g))
    while all(v == 1):  # Mutate until a change occurs (prevent duplicates)
        v = (g * (npr.random(len(g)) < mp) * npr.randn(len(g)) * npr.random() * s + 1).clip(0.3, 3.0)
    for i, k in enumerate(hyp.keys()):
        hyp[k] = x[i + 7] * v[i]  # Mutate


def clip_hyperparameters(hyp, limits):
    keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
    for k, v in zip(keys, limits):
        hyp[k] = np.clip(hyp[k], v[0], v[1])






def train(opt):
    def initialize_optimizer(model, opt, hyp):
        # Optimizer parameter groups
        pg0, pg1, pg2 = [], [], []
        for k, v in dict(model.named_parameters()).items():
            if '.bias' in k:
                pg2.append(v)  # biases
            elif 'Conv2d.weight' in k:
                pg1.append(v)  # apply weight_decay
            else:
                pg0.append(v)  # all else

        optimizer = optim.Adam(pg0, lr=hyp['lr0']) if opt.adam else optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        return optimizer


    def initialize_scheduler(optimizer, epochs):
        # Cosine scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=-1)
        return scheduler, lf


    def initialize_distributed_training(model, device):
        if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
            dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:9999', world_size=1, rank=0)
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level
        return model


    def setup_training(opt, hyp, device, yolo_props, freeze):
       
        data = opt.data
        weights = opt.weights
        imgsz_min, imgsz_max, imgsz_test = opt.img_size

        # Image Sizes
        gs = 64
        assert math.fmod(imgsz_min, gs) == 0, f'--img-size {imgsz_min} must be a {gs}-multiple'
        opt.multi_scale |= imgsz_min != imgsz_max
        if opt.multi_scale:
            if imgsz_min == imgsz_max:
                imgsz_min //= 1.5
                imgsz_max //= 0.667
            grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
            imgsz_min, imgsz_max = grid_min * gs, grid_max * gs
        img_size = imgsz_max

        # Configure run
        init_seeds()
        data_dict = parse_data_cfg(data)        
        nc = 1 if opt.single_cls else int(data_dict['classes'])
        hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

        # Remove previous results
        for f in glob.glob('*_batch*.png') + glob.glob(RESULTS_FILE):
            os.remove(f)

        # Initialize model
        model = MDENet(path=weights, yolo_props=yolo_props, freeze=freeze).to(device)

        # Initialize optimizer, scheduler, and distributed training
        optimizer = initialize_optimizer(model, opt, hyp)
        scheduler, lf = initialize_scheduler(optimizer, epochs)
        model = initialize_distributed_training(model, device)

        # Mixed precision training setup
        if mixed_precision:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

        return model, optimizer, scheduler, nc, hyp, lf, img_size, gs, grid_min, grid_max
    
    def create_dataset(train_path, img_size, batch_size, hyp, opt, freeze):
        return LoadImagesAndLabels(train_path, img_size, batch_size,
                               augment=not freeze["yolo"],
                               hyp=hyp,  # augmentation hyperparameters
                               rect=opt.rect,  # rectangular training
                               cache_images=opt.cache_images,
                               single_cls=opt.single_cls)


    def create_dataloader(dataset, batch_size):
        batch_size = min(batch_size, len(dataset))
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        return torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        num_workers=nw,
                                        shuffle=True,  # Shuffle=True unless rectangular training is used
                                        pin_memory=True,
                                        collate_fn=dataset.collate_fn), nw


    def create_testloader(test_path, imgsz_test, batch_size, hyp, opt, dataset):
        return torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                            hyp=hyp,
                                                            rect=True,
                                                            cache_images=opt.cache_images,
                                                            single_cls=opt.single_cls),
                                        batch_size=batch_size,
                                        num_workers=dataset.num_workers,
                                        pin_memory=True,
                                        collate_fn=dataset.collate_fn)


    def setup_dataset_and_dataloader(train_path, img_size, batch_size, hyp, opt, freeze, test_path, imgsz_test):        
        print(f"train_path: {train_path}")
        print(f"img_size: {img_size}")
        print(f"batch_size: {batch_size}")
        print(f"hyp: {hyp}")
        print(f"opt.rect: {opt.rect}")
        print(f"opt.cache_images: {opt.cache_images}")
        print(f"opt.single_cls: {opt.single_cls}")

        dataset = create_dataset(train_path, img_size, batch_size, hyp, opt, freeze)
        print(f"Dataloader batch_size: {batch_size}")
        print(f"Number of workers: {os.cpu_count() if batch_size > 1 else 0}")
        print(f"opt.rect: {opt.rect}")
        print(f"dataset.collate_fn: {dataset.collate_fn}")

        dataloader, nw = create_dataloader(dataset, batch_size)
        testloader = create_testloader(test_path, imgsz_test, batch_size, hyp, opt, dataset)        
        return dataset, dataloader, testloader, nw

    def handle_burn_in(ni, n_burn, model, optimizer, lf, hyp):
        if ni <= n_burn * 2:
            model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])
            if ni == n_burn:
                print_model_biases(model)

            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf()])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])


    def handle_multi_scale_training(opt, ni, accumulate, imgs, gs, grid_min, grid_max):
        if opt.multi_scale:
            if ni / accumulate % 1 == 0:
                img_size = random.randrange(grid_min, grid_max + 1) * gs
            sf = img_size / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
        return imgs


    def update_image_weights(model, dataset, maps, nc):
        w = model.class_weights.cpu().numpy() * (1 - maps) ** 2
        image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
        dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)


    def process_batch(ni,n_burn, imgs, targets, model, opt, accumulate, gs, grid_min, grid_max, device):
        imgs = imgs.to(device).float() / 255.0
        targets = targets.to(device)
        handle_burn_in(ni, n_burn, model, optimizer, lf, hyp)
        imgs = handle_multi_scale_training(opt, ni, accumulate, imgs, gs, grid_min, grid_max)
        midas_out, yolo_out = model(imgs)
        return midas_out, yolo_out, imgs, targets


    def main_training_loop(epochs, start_epoch, best_fitness, model, optimizer, dataset, dataloader, scheduler, device, tb_writer):
        nb = len(dataloader)
        n_burn = max(3 * nb, 500)
        maps = np.zeros(nc)
        results = (0, 0, 0, 0, 0, 0, 0)
        t0 = time.time()

        for param in model.pretrained.parameters():
            param.requires_grad = False

        print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
        print('Using %g dataloader workers' % nw)
        print('Starting training for %g epochs...' % epochs)

        for epoch in range(start_epoch, epochs):
            model.train()
            if dataset.image_weights:
                update_image_weights(model, dataset, maps, nc)

            mloss = torch.zeros(5).to(device)
            print(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'l_depth', 'total', 'targets', 'img_size'))
            pbar = tqdm(enumerate(dataloader), total=nb)
            for i, (imgs, targets, paths, _, dp_imgs, pln_imgs) in pbar:
                ni = i + nb * epoch
                midas_out, yolo_out, imgs, targets = process_batch(ni, n_burn, imgs, targets, model, opt, accumulate, gs, grid_min, grid_max, device)
                
                # Compute loss
                loss, loss_items = compute_loss(yolo_out, targets, midas_out, dp_imgs, alpha, model)
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Scale loss by nominal batch_size of 64
                loss *= batch_size / 64

                # Compute gradient
                if mixed_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Optimize accumulated gradient
                if ni % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update(model)

                # Print batch results
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                s = ('%10s' * 2 + '%10.3g' * 7) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
                pbar.set_description(s)

                # Plot images with bounding boxes
                if ni < 1:
                    f = 'train_batch%g.png' % i
                    plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer:
                        tb_writer.add_image(f, cv2.imread(f)[:, :, ::-1], dataformats='HWC')

            # Update scheduler
            scheduler.step()

            # Process epoch results
            ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:
                is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
                results, maps = test.test(cfg,
                                        data,
                                        batch_size=batch_size,
                                        img_size=imgsz_test,
                                        model=ema.ema,
                                        save_json=final_epoch and is_coco,
                                        single_cls=opt.single_cls,
                                        dataloader=testloader)

            # Write epoch results
            with open(RESULTS_FILE, 'a') as f:
                f.write(s + '%10.3g' * 8 % results + '\n')
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

            # Write Tensorboard results
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi

            # Save training results
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(RESULTS_FILE, 'r') as f:
                    chkpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                torch.save(chkpt, LAST_MODEL)

                if (best_fitness == fi) and not final_epoch:
                    torch.save(chkpt, BEST_MODEL)

                del chkpt

        return results, epoch, t0

    def rename_and_upload_files(opt, wdir, results_file, epoch, start_epoch, t0):
        n = opt.name
        if len(n):
            n = f'_{n}' if not n.isnumeric() else n
            fresults, flast, fbest = f'results{n}.txt', os.path.join(wdir, f'last{n}.pt'), os.path.join(wdir, f'best{n}.pt')
            files_to_rename = [
                (os.path.join(wdir, 'last.pt'), flast),
                (os.path.join(wdir, 'best.pt'), fresults),
                ('results.txt', fresults)
            ]
            for src, dest in files_to_rename:
                if os.path.exists(src):
                    os.rename(src, dest)
                    if dest.endswith('.pt'):
                        strip_optimizer(dest)
                        if opt.bucket:
                            os.system(f'gsutil cp {dest} gs://{opt.bucket}/weights')

        if not opt.evolve:
            plot_results()
        print(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
        if torch.cuda.device_count() > 1:
            dist.destroy_process_group()
        torch.cuda.empty_cache()

    #Traing setup
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)


    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']

    model, optimizer, scheduler, nc, hyp, lf, img_size, gs, grid_min, grid_max = setup_training(device=device, yolo_props=yolo_props, freeze=freeze)
    dataset, dataloader, testloader, nw = setup_dataset_and_dataloader(train_path, img_size, batch_size, hyp, opt, freeze, test_path, imgsz_test)

    
    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    start_epoch = 0
    best_fitness = 0.0

    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # Start training
    results, epoch, t0 = main_training_loop(epochs, start_epoch, best_fitness, model, optimizer, dataset, dataloader, scheduler, device, tb_writer)    
    # end training
    
    rename_and_upload_files(opt, WEIGHTS_DIR, RESULTS_FILE, epoch, start_epoch, t0)


    return results

    
def main(params):
    
    #Getting received parameters
    opt = ConfigNamespace(params)
    print(opt)

    #Defining initial values for global values
    global mixed_precision
    mixed_precision = setup_mixed_precision()

    global hyp
    hyp = load_hyperparameters()    
        
    global conf
    conf = read_config()

    global yolo_props
    yolo_props = get_yolo_properties(conf)
    
    global freeze, alpha 
    freeze, alpha = get_freeze_alpha_values(conf) 

    global device
    device = select_device_and_precision(opt)    
    
    global tb_writer
    tb_writer = initialize_tensorboard(opt)

    #Altering based on parameter values
    log_focal_loss(hyp)
    opt.weights = 'last.pt' if opt.resume else opt.weights
    extend_img_size(opt)
    

    if not opt.evolve:
        train(opt)  # Train normally
    else:
        opt.notest, opt.nosave = True, True  # Only test/save final epoch
        download_evolve_file(opt)
        
        for _ in range(1):  # Generations to evolve
            if os.path.exists('evolve.txt'):
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # Number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # Top n mutations
                parent = select_parent(x, n, fitness)
                mutate_hyperparameters(parent, hyp)
                clip_hyperparameters(hyp, [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)])

                results = train()  # Train mutation
                print_mutation(hyp, results, opt.bucket)
                # plot_evolution_results(hyp)
    