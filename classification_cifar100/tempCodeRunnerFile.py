                with open(gradient_file_path, 'a') as f:
                    writer = csv.writer(f)
                    for gradient_tensor in gradients:
                        writer.writerow(gradient_tensor.cpu().numpy())