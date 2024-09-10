from mmdet.apis import init_detector, inference_detector

def main():
    config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
    checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
    print(inference_detector(model, 'demo.jpg'))

if __name__ == "__main__":
    main()

