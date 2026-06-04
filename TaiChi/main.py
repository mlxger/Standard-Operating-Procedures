"""
TaiChi 24式动作识别系统 — 统一入口

python main.py extract                              # Step 1: 提取关键点
python main.py train                                # Step 2: 训练模型
python main.py realtime --video path/to/video.mp4  # Step 3: 视频推理
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='TaiChi 24式动作识别系统 (MediaPipe Tasks 0.10.x)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'mode',
        choices=['extract', 'train', 'realtime'],
        help=(
            'extract  : 从帧图片提取 MediaPipe 关键点\n'
            'train    : 训练 TaiChiTransformer\n'
            'realtime : 对视频文件进行实时动作识别'
        )
    )
    parser.add_argument(
        '--video',
        default=None,
        help='[realtime 模式必填] 视频文件路径'
    )
    parser.add_argument(
        '--model',
        default='./checkpoints/taichi_transformer_best.pth',
        help='[realtime 模式] 模型权重路径（默认: ./checkpoints/taichi_transformer_best.pth）'
    )
    args = parser.parse_args()

    # ── Step 1: 提取关键点 ─────────────────────────────────────────
    if args.mode == 'extract':
        print("━" * 55)
        print("  Step 1 │ 提取 MediaPipe Pose 关键点")
        print("━" * 55)
        from extract_keypoints import extract_all
        extract_all()

    # ── Step 2: 训练 ───────────────────────────────────────────────
    elif args.mode == 'train':
        print("━" * 55)
        print("  Step 2 │ 训练 TaiChiTransformer")
        print("━" * 55)
        from train import train
        train()

    # ── Step 3: 视频推理 ────────────────────────────────────────────
    elif args.mode == 'realtime':
        if args.video is None:
            print("[错误] realtime 模式需要通过 --video 指定视频路径")
            print("示例: python main.py realtime --video taichi.mp4")
            sys.exit(1)

        print("━" * 55)
        print("  Step 3 │ 视频实时推理")
        print("━" * 55)
        from realtime import TaiChiPredictor
        predictor = TaiChiPredictor(model_path=args.model)
        predictor.run(video_path=args.video)


if __name__ == "__main__":
    main()