def save_checkpoint(optimizer,optimizer_opt, metrics_history,epoch_size, filename="checkpoint.pth"):
    checkpoint = {
        'epoch_size': epoch_size,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_opt_state_dict': optimizer_opt.state_dict(),
        'metrics': metrics_history,
        
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")



def load_checkpoint(PATH):
    # Load weights and optimizer states
    checkpoint = torch.load(PATH, weights_only=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer_opt.load_state_dict(checkpoint['optimizer_opt_state_dict'])
    epoch_size = checkpoint['epoch_size']
    
    metrics = checkpoint['metrics']
