from safetensors import safe_open

file_path = "D:\\img2latex\\latex_ocr\\model.safetensors"

with safe_open(file_path, framework="pt", device="cuda") as f:
    metadata = f.metadata()
    if metadata:
        print("--- Metadata của mô hình ---")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    else:
        print("File này không chứa metadata.")

    print("\n" + "="*75)

    tensor_names = f.keys()
    print(f"{'Tên Tensor':<50} | {'Kích thước (Shape)':<20}")
    print("-" * 75)
    
    for name in tensor_names:
        tensor_slice = f.get_slice(name)
        print(f"{name:<50} | {str(tensor_slice.get_shape()):<20}")
