import os
import glob
import nibabel as nib
import numpy as np
import shutil
from nipype.interfaces.ants import N4BiasFieldCorrection
# from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.utils import shuffle
import scipy.misc

num_mod = 2

def get_filename(set_name, case_idx, input_name, loc):
    pattern = '{0}/{1}/{3}/subject-{2}-{3}.nii'
    return pattern.format(loc, set_name, case_idx, input_name)

def get_set_name(case_idx):
    return 'Training' if case_idx < 5 else 'Testing'

def read_data(case_idx, input_name, loc):
    set_name = get_set_name(case_idx)
    image_path = get_filename(set_name, case_idx, input_name, loc)
    print(image_path)
    return nib.load(image_path)

from scipy.ndimage import zoom

def resize_volume(volume, target_shape):
    factors = [
        t / s for t, s in zip(target_shape, volume.shape)
    ]
    return zoom(volume, factors, order=1)  # Order 1 = linear interpolation

def read_vol(case_idx, input_name, loc, target_shape=(144, 192, 256)):
    image_data = read_data(case_idx, input_name, loc)
    volume = image_data.get_fdata()
    if volume.shape != target_shape:
        volume = resize_volume(volume, target_shape)
    return volume

def correct_bias(in_file, out_file):
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    done = correct.run()
    return done.outputs.output_image

def normalise(case_idx, input_name, in_dir, out_dir,copy=False):
	set_name = get_set_name(case_idx)
	image_in_path = get_filename(set_name, case_idx, input_name, in_dir)
	image_out_path = get_filename(set_name, case_idx, input_name, out_dir)
	if copy:
		shutil.copy(image_in_path, image_out_path)
	else:
		correct_bias(image_in_path, image_out_path)
	print(image_in_path + " done.")

"""
To extract patches from a 3D image
"""
def extract_patches(volume, patch_shape, extraction_step,datype='float32'):
    patch_h, patch_w, patch_d = patch_shape[0], patch_shape[1], patch_shape[2]
    stride_h, stride_w, stride_d = extraction_step[0], extraction_step[1], extraction_step[2]
    img_h, img_w, img_d = volume.shape[0],volume.shape[1],volume.shape[2]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_d = (img_d-patch_d)//stride_d+1
    N_patches_img = N_patches_h * N_patches_w * N_patches_d
    raw_patch_martrix = np.zeros((N_patches_img,patch_h,patch_w,patch_d),dtype=datype)
    k=0

    #iterator over all the patches
    for h in range((img_h-patch_h)//stride_h+1):
        for w in range((img_w-patch_w)//stride_w+1):
            for d in range((img_d-patch_d)//stride_d+1):
                raw_patch_martrix[k]=volume[h*stride_h:(h*stride_h)+patch_h,\
                                			w*stride_w:(w*stride_w)+patch_w,\
                                  				d*stride_d:(d*stride_d)+patch_d]
                k+=1
    assert(k==N_patches_img)
    return raw_patch_martrix


"""
To extract labeled patches from array of 3D labeled images
"""
def get_patches_lab(T1_vols, T2_vols, label_vols, extraction_step,
                    patch_shape,validating,testing,num_images_training):
    patch_shape_1d=patch_shape[0]
    # Extract patches from input volumes and ground truth
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, num_mod),dtype="float32")
    y = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d),dtype="uint8")
    for idx in range(len(T1_vols)) :
        y_length = len(y)
        if testing:
            print(("Extracting Patches from Image %2d ....")%(num_images_training+idx+2))
        elif validating:
            print(("Extracting Patches from Image %2d ....")%(num_images_training+idx+1))
        else:
            print(("Extracting Patches from Image %2d ....")%(1+idx))
        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step,
        														datype="uint8")

        # Select only those who are important for processing
        if testing or validating:
            valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != -1)
        else:
            valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2, 3)) > 100)

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d,
                                                patch_shape_1d, patch_shape_1d, num_mod),dtype="float32")))
        y = np.vstack((y, np.zeros((len(label_patches), patch_shape_1d,
                                                patch_shape_1d, patch_shape_1d),dtype="uint8")))

        y[y_length:, :, :, :] = label_patches

        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step,datype="float32")
        x[y_length:, :, :, :, 0] = T1_train[valid_idxs]

        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step,datype="float32")
        x[y_length:, :, :, :, 1] = T2_train[valid_idxs]
    return x, y

"""
To preprocess the labeled training data
"""
def preprocess_dynamic_lab(dir, seed, num_classes, extraction_step, patch_shape, validating, testing, num_images_training=2, num_images_testing=7):
    x = list(range(5, 9))
    if testing:
        print("Testing")
        index_start = num_images_training + 2
        index_end = index_start + num_images_testing
        T1_vols = np.empty((num_images_testing, 144, 192, 256), dtype="float32")
        T2_vols = np.empty((num_images_testing, 144, 192, 256), dtype="float32")
        label_vols = np.empty((num_images_testing, 144, 192, 256), dtype="uint8")
    elif validating:
        print("Validating")
        index_start = num_images_training + 1
        index_end = index_start + 1
        T1_vols = np.empty((1, 144, 192, 256), dtype="float32")
        T2_vols = np.empty((1, 144, 192, 256), dtype="float32")
        label_vols = np.empty((1, 144, 192, 256), dtype="uint8")
    else:
        print("Training")
        index_start = 1
        index_end = index_start + num_images_training
        T1_vols = np.empty((num_images_training, 144, 192, 256), dtype="float32")
        T2_vols = np.empty((num_images_training, 144, 192, 256), dtype="float32")
        label_vols = np.empty((num_images_training, 144, 192, 256), dtype="uint8")
    i = 0
    for index in range(index_start, min(index_end, len(x) + 5)):
        print(x[index-5])
        T1_vols[i, :, :, :] = read_vol(x[index-5], 'T1', dir)
        T2_vols[i, :, :, :] = read_vol(x[index-5], 'T2', dir)
        label_vols[i, :, :, :] = read_vol(x[index-5], 'label', dir)
        i = i + 1

    # Standardization
    T1_mean = T1_vols.mean()
    T1_std = T1_vols.std()
    T1_vols = (T1_vols - T1_mean) / T1_std
    T2_mean = T2_vols.mean()
    T2_std = T2_vols.std()
    T2_vols = (T2_vols - T2_mean) / T2_std

    # Normalization with Conditional Check
    for i in range(T1_vols.shape[0]):
        min_val = np.min(T1_vols[i])
        max_val = np.max(T1_vols[i])
        denominator = max_val - min_val
        if denominator != 0:
            T1_vols[i] = ((T1_vols[i] - min_val) / denominator) * 255
        else:
            print(f"Warning: T1_vols[{i}] has zero denominator. Setting to zeros.")
            T1_vols[i] = np.zeros_like(T1_vols[i])

    for i in range(T2_vols.shape[0]):
        min_val = np.min(T2_vols[i])
        max_val = np.max(T2_vols[i])
        denominator = max_val - min_val
        if denominator != 0:
            T2_vols[i] = ((T2_vols[i] - min_val) / denominator) * 255
        else:
            print(f"Warning: T2_vols[{i}] has zero denominator. Setting to zeros.")
            T2_vols[i] = np.zeros_like(T2_vols[i])

    # Scale to [-1, 1]
    T1_vols = T1_vols / 127.5 - 1.
    T2_vols = T2_vols / 127.5 - 1.

    x, y = get_patches_lab(
        T1_vols, T2_vols, label_vols, extraction_step, patch_shape,
        validating=validating, testing=testing, num_images_training=num_images_training
    )
    print("Total Extracted Labelled Patches Shape:", x.shape, y.shape)

    if testing:
        return np.rollaxis(x, 4, 1), label_vols
    elif validating:
        return np.rollaxis(x, 4, 1), y, label_vols
    else:
        return np.rollaxis(x, 4, 1), y

"""
To extract labeled patches from array of 3D ulabeled images
"""
def get_patches_unlab(T1_vols, T2_vols, extraction_step, patch_shape, dir):
    patch_shape_1d = patch_shape[0]
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, num_mod))

    for idx in range(len(T1_vols)):
        # print(f"Processing Image {idx + 11} ...")
        # print(f"T1_vols[{idx}] shape: {T1_vols[idx].shape}")
        # print(f"T2_vols[{idx}] shape: {T2_vols[idx].shape}")

        # Validate patch dimensions
        if (
            T1_vols[idx].shape[0] < patch_shape[0] or
            T1_vols[idx].shape[1] < patch_shape[1] or
            T1_vols[idx].shape[2] < patch_shape[2]
        ):
            print(f"Skipping Image {idx + 11} due to incompatible dimensions.")
            continue

        # Extract patches with error handling
        try:
            print(f"Extracting patches with Patch shape: {patch_shape}, Extraction step: {extraction_step}")
            T1_patches = extract_patches(T1_vols[idx], patch_shape, extraction_step, datype="float32")
            T2_patches = extract_patches(T2_vols[idx], patch_shape, extraction_step, datype="float32")
        except Exception as e:
            print(f"Error extracting patches for Image {idx + 11}: {e}")
            continue

        # Allow all patches for unlabeled data
        valid_idxs = np.arange(T1_patches.shape[0])

        x = np.vstack((x, np.zeros((len(valid_idxs), patch_shape_1d, patch_shape_1d, patch_shape_1d, num_mod))))
        x[-len(valid_idxs):, :, :, :, 0] = T1_patches[valid_idxs]
        x[-len(valid_idxs):, :, :, :, 1] = T2_patches[valid_idxs]

    print(f"Total patches extracted: {x.shape}")
    return x

"""
To preprocess the unlabeled training data
"""
def preprocess_dynamic_unlab(dir, extraction_step, patch_shape, num_images_training_unlab):
    T1_vols = np.empty((num_images_training_unlab, 144, 192, 256), dtype="float32")
    T2_vols = np.empty((num_images_training_unlab, 144, 192, 256), dtype="float32")
    for case_idx in range(11, 11 + num_images_training_unlab):
        T1_vols[(case_idx - 11), :, :, :] = read_vol(case_idx, 'T1', dir)
        T2_vols[(case_idx - 11), :, :, :] = read_vol(case_idx, 'T2', dir)
        # print(read_vol(case_idx, 'T2', dir).shape)

    # Standardization
    T1_mean = T1_vols.mean()
    T1_std = T1_vols.std()
    T1_vols = (T1_vols - T1_mean) / T1_std
    T2_mean = T2_vols.mean()
    T2_std = T2_vols.std()
    T2_vols = (T2_vols - T2_mean) / T2_std

    # Normalization with Conditional Check
    for i in range(T1_vols.shape[0]):
        min_val = np.min(T1_vols[i])
        max_val = np.max(T1_vols[i])
        denominator = max_val - min_val
        if denominator != 0:
            T1_vols[i] = ((T1_vols[i] - min_val) / denominator) * 255
        else:
            print(f"Warning: T1_vols[{i}] has zero denominator. Setting to zeros.")
            T1_vols[i] = np.zeros_like(T1_vols[i])

    for i in range(T2_vols.shape[0]):
        min_val = np.min(T2_vols[i])
        max_val = np.max(T2_vols[i])
        denominator = max_val - min_val
        if denominator != 0:
            T2_vols[i] = ((T2_vols[i] - min_val) / denominator) * 255
        else:
            print(f"Warning: T2_vols[{i}] has zero denominator. Setting to zeros.")
            T2_vols[i] = np.zeros_like(T2_vols[i])

    # Scale to [-1, 1]
    T1_vols = T1_vols / 127.5 - 1.
    T2_vols = T2_vols / 127.5 - 1.

    x = get_patches_unlab(T1_vols, T2_vols, extraction_step, patch_shape, dir)
    print("Total Extracted Unlabeled Patches Shape:", x.shape)
    return np.rollaxis(x, 4, 1)