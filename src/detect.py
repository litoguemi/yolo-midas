import argparse
import configparser as cp
from sys import platform

from src.model.mde_net import *  # set ONNX_EXPORT in models.py
from src.model.monocular_depth_estimation_dataset import *
from src.utils.utils import *

# Configuration setup
conf_path = "cfg/mde.cfg"
conf = cp.RawConfigParser()
conf.read(conf_path)

def get_yolo_properties(conf):
    yolo_props = {
        "anchors": np.array([float(x) for x in conf.get("yolo", "anchors").split(',')]).reshape((-1, 2)),
        "num_classes": conf.get("yolo", "classes")
    }
    return yolo_props

def get_freeze_alpha_values(conf):
    layers = ["resnet", "midas", "yolo", "planercnn"]
    freeze, alpha = {}, {}
    for layer in layers:
        freeze[layer], alpha[layer] = (True, 0) if conf.get("freeze", layer) == "True" else (False, 1)
    return freeze, alpha


def initialize_device(opt):
    return torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

def initialize_model(opt, yolo_props, freeze, device):
    model = MDENet(path=opt.weights, yolo_props=yolo_props, freeze=freeze).to(device)
    model.eval()
    return model

def set_dataloader(opt, img_size, source, save_img):
    if source == '0' or source.startswith(('rtsp', 'http')) or source.endswith('.txt'):
        opt.view_img = True
        torch.backends.cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)
    return dataset, save_img

def get_names_colors(names_path):
    names = load_classes(names_path)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    return names, colors

def run_inference(model, img, half, device):
    img = torch.from_numpy(img)
    img = img.to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    depth_pred, pred, _ = model(img, augment=False)
    return depth_pred, pred

def detect(opt):    
    global conf
    yolo_props = get_yolo_properties(conf)
    freeze, alpha = get_freeze_alpha_values(conf)
    device = initialize_device(opt)
    model = initialize_model(opt, yolo_props, freeze, device)

    if opt.half:
        model.half()

    dataset, save_img = set_dataloader(opt, (320, 192) if ONNX_EXPORT else opt.img_size, opt.source, False)
    names, colors = get_names_colors(opt.names)

    t0 = time.time()
    if device.type != 'cpu':
        model(torch.zeros((1, 3, opt.img_size, opt.img_size), device=device))

    for path, img, im0s, vid_cap in dataset:
        depth_pred, pred = run_inference(model, img, opt.half, device)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
        
        # Process detections
        for i, det in enumerate(pred):
            p, s, im0 = (path[i], '%g: ' % i, im0s[i]) if dataset.mode == 'images' else (path, '', im0s)
            save_path = str(Path(opt.output) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])
                for *xyxy, conf, cls in det:
                    if opt.save_txt:
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                    if save_img or opt.view_img:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
            print('%sDone. (%.3fs)' % (s, time.time() - t0))

            if opt.view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

        # Save depth predictions
        for i, depth in enumerate(depth_pred):
            prediction = depth.squeeze().cpu().numpy()
            filename = save_path.split(".")[0] + "_depth." + save_path.split(".")[1]
            write_depth(filename, prediction, bits=2)

    if opt.save_txt or save_img:
        print(f'Results saved to {os.getcwd()}{os.sep}{opt.output}')
        if platform == 'darwin':
            os.system('open ' + opt.output + ' ' + save_path)

    print(f'Done. ({time.time() - t0:.3f}s)')


def main(params):
    #Getting received parameters    
    opt = ConfigNamespace(params)
    with torch.no_grad():
        detect(opt)
    