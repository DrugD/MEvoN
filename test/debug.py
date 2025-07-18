import torch

def occupy_40gb_vram():
    # 检查是否有可用的CUDA设备
    if not torch.cuda.is_available():
        print("CUDA不可用，无法占用显存")
        return
    
    device = torch.device('cuda:3')
    
    # 计算需要分配的张量大小（每个float32占4字节）
    # 40GB = 40 * 1024^3 bytes
    bytes_needed = 40 * 1024 ** 3
    elements_needed = bytes_needed // 4  # 每个float32元素占4字节
    
    # 计算张量形状，尽量接近正方形以减少维度数量
    dim = int(elements_needed ** 0.5)
    remaining = elements_needed - dim * dim
    
    print(f"尝试分配约40GB显存...")
    
    try:
        # 创建一个大张量
        big_tensor1 = torch.zeros((dim, dim), dtype=torch.float32, device=device)
        
        # 如果有剩余元素，再创建一个小张量
        if remaining > 0:
            big_tensor2 = torch.zeros((remaining,), dtype=torch.float32, device=device)
        
        print(f"成功占用约40GB显存")
        print("按Ctrl+C停止并释放显存")
        
        # 保持占用状态
        while True:
            pass
            
    except RuntimeError as e:
        print(f"分配失败: {e}")
    except KeyboardInterrupt:
        print("\n释放显存...")

if __name__ == "__main__":
    occupy_40gb_vram()