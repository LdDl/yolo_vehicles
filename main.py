from pathlib import Path
import shutil

def replace_class_with_custom_id(class_id):
    if class_id == 0: # AIC HCMC 2020 motorbike
        return 1 # replace with motorbike
    if class_id == 1: # AIC HCMC 2020 car
        return 0 # replace with car
    if class_id == 2: # AIC HCMC 2020 bus
        return 2 # replace with bus
    if class_id == 3: # AIC HCMC 2020 truck
        return 3 # replace with truck

def modify_labels(f, newf):
    newfname = newf / f.name
    with open(f) as _f:
        boxes = _f.readlines()
        newboxes = []
        for box in boxes:
            yolo_annotation = box.rstrip('\r\n').split(" ")
            class_id = int(yolo_annotation[0])
            class_id_new = replace_class_with_custom_id(class_id)
            yolo_annotation[0] = "{}".format(class_id_new)
            newboxes.append(" ".join(yolo_annotation))
        with open(newfname, 'w') as _newf:
            contents = "\n".join(newboxes)
            _newf.writelines(contents)

def modify_data(dir, dir_custom):
    try:
        dir_custom.mkdir(parents=True, exist_ok=False)
    except:
        print(dir_custom, 'already exists')
    for label_file in dir.iterdir():
        modify_labels(label_file, dir_custom)

def prepare_pathes(images_dir, source_dir, source_file, output_file):
    print("Preparing TXT file with abosolute paths for", source_file)
    file = source_dir / source_file
    data = []
    print(source_dir, source_file)
    with open(file) as _file:
        data = _file.readlines()
    prepared_names = []
    for row in data:
        row_data = row.rstrip('\r\n').split(" ")
        fname = row_data[0]
        prepared_names.append("{}".format(images_dir.resolve() / fname))

    out_file = source_dir / output_file
    with open(out_file, "w") as _out_file:
        file_contents = "\n".join(prepared_names)
        _out_file.writelines(file_contents)

def prepare_absolute_pathes(images_dir, source_dir, source_file, output_file):
    print("Preparing TXT file with abosolute paths for", source_file)
    file = source_dir / source_file
    data = []
    print(source_dir, source_file)
    with open(file) as _file:
        data = _file.readlines()
    prepared_names = []
    for row in data:
        row_data = row.rstrip('\r\n').split(" ")
        fname = row_data[0]
        prepared_names.append("{}".format(images_dir.resolve() / fname))

    out_file = source_dir / output_file
    with open(out_file, "w") as _out_file:
        file_contents = "\n".join(prepared_names)
        _out_file.writelines(file_contents)

def copy_labels_to_images(images, labels):
    for label_file in labels.iterdir():
        # For Python <= 3.7
        # shutil.copy(str(my_file), str(to_file))  
        # For Python 3.8+
        shutil.copy(label_file, images / label_file.name) 

def main():
    print("Start AIC HCMC 20202 Challenge data preparation")
    base_dir = Path("./aic_hcmc2020/aic_hcmc2020")
    images_dir = base_dir / 'images'
    labels_dir = base_dir / 'labels'
    labels_custom_dir = base_dir / 'labels_custom'

    modify_data(labels_dir, labels_custom_dir)
    copy_labels_to_images(images_dir, labels_custom_dir)
    current_dir = Path(".")
    train_file = 'train_aic_hcmc_relative.txt'
    save_train_file = 'train_aic_hcmc.txt'
    prepare_absolute_pathes(images_dir, current_dir, train_file, save_train_file)
    val_file = 'val_aic_hcmc_relative.txt'
    save_val_file = 'val_aic_hcmc.txt'
    prepare_absolute_pathes(images_dir, current_dir, val_file, save_val_file)

if __name__ == "__main__":
    main()