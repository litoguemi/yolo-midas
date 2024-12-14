import argparse
import os
import configparser

def main():

    config = configparser.ConfigParser() 
    config.read('cfg/config.ini')

    parser = argparse.ArgumentParser(description="Main entrance for the application.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Create parser for the "train" command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='cfg/config.ini', help='Path to configuration file')

    # Create parser for the "detect" command
    detect_parser = subparsers.add_parser('detect', help='Detect objects')
    detect_parser.add_argument('--config', type=str, default='cfg/config.ini', help='Path to configuration file')

    args = parser.parse_args()

    config = configparser.ConfigParser() 
    config.read(args.config)

    if args.command == 'train':
        train_params = {key: value for key, value in config['train'].items()}
        import src.train as tr
        tr.main(train_params)
    elif args.command == 'detect':
        detect_params = {key: value for key, value in config['detect'].items()}
        import src.detect as dt
        dt.main(detect_params)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
