"""
主执行脚本 - 框架偏见检测模型的入口点
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from framing_bias_detector.config import Config, load_config, default_config
from framing_bias_detector.experiment_manager import ExperimentManager
from framing_bias_detector.utils import setup_environment, check_requirements

logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="框架偏见检测模型训练和评估")
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'predict', 'experiment'],
                       help='运行模式')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据文件路径')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数量')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='训练轮数')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default=None,
                       help='预训练模型路径')
    parser.add_argument('--fusion_method', type=str, default=None,
                       choices=['concat', 'weighted_sum', 'attention'],
                       help='特征融合方法')
    
    # 实验参数
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    parser.add_argument('--no_wandb', action='store_true',
                       help='禁用wandb日志')
    
    return parser.parse_args()

def update_config_from_args(config: Config, args):
    """根据命令行参数更新配置"""
    
    # 数据配置
    if args.data_path:
        config.data.data_path = args.data_path
    if args.max_samples:
        config.data.max_samples = args.max_samples
    
    # 训练配置
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    
    # 模型配置
    if args.fusion_method:
        config.model.fusion_method = args.fusion_method
    
    # 实验配置
    if args.experiment_name:
        config.experiment.experiment_name = args.experiment_name
    if args.output_dir:
        config.experiment.output_dir = args.output_dir
    if args.seed:
        config.experiment.seed = args.seed
    if args.device:
        config.experiment.device = args.device
    if args.no_wandb:
        config.experiment.use_wandb = False
    
    return config

def main():
    """主函数"""
    
    # 解析参数
    args = parse_arguments()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("启动框架偏见检测模型...")
    logger.info(f"运行模式: {args.mode}")
    
    try:
        # 检查环境和依赖
        setup_environment()
        check_requirements()
        
        # 加载配置
        if args.config:
            logger.info(f"从文件加载配置: {args.config}")
            config = load_config(args.config)
        else:
            logger.info("使用默认配置")
            config = default_config
        
        # 根据命令行参数更新配置
        config = update_config_from_args(config, args)
        
        # 验证数据路径
        if not os.path.exists(config.data.data_path):
            logger.error(f"数据文件不存在: {config.data.data_path}")
            sys.exit(1)
        
        # 创建实验管理器
        experiment_manager = ExperimentManager(config)
        
        # 根据模式执行不同操作
        if args.mode == 'experiment':
            # 运行完整实验
            logger.info("开始运行完整实验...")
            results = experiment_manager.run_full_experiment()
            
            logger.info("实验完成!")
            logger.info(f"实验目录: {experiment_manager.experiment_dir}")
            
            # 打印主要结果
            if 'test_metrics' in results:
                test_metrics = results['test_metrics']
                logger.info("测试结果:")
                logger.info(f"  准确率: {test_metrics.get('accuracy', 'N/A'):.4f}")
                logger.info(f"  F1分数: {test_metrics.get('f1_weighted', 'N/A'):.4f}")
                logger.info(f"  AUC: {test_metrics.get('auc', 'N/A'):.4f}")
        
        elif args.mode == 'train':
            # 仅训练模式
            logger.info("开始训练模式...")
            experiment_manager.initialize_components()
            train_loader, val_loader, test_loader, test_dataset = experiment_manager.prepare_data()
            experiment_manager.run_training(train_loader, val_loader)
            
        elif args.mode == 'eval':
            # 仅评估模式
            logger.info("开始评估模式...")
            if not args.model_path:
                logger.error("评估模式需要指定模型路径 --model_path")
                sys.exit(1)
            
            experiment_manager.initialize_components()
            _, _, test_loader, test_dataset = experiment_manager.prepare_data()
            
            # 加载模型
            experiment_manager.trainer.load_checkpoint(args.model_path)
            
            # 评估
            test_metrics = experiment_manager.run_evaluation(test_loader, test_dataset)
            logger.info(f"评估完成: {test_metrics}")
        
        elif args.mode == 'predict':
            # 预测模式
            logger.info("预测模式暂未实现")
            # TODO: 实现预测模式
        
        # 清理资源
        experiment_manager.cleanup()
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()