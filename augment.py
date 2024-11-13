import argparse
from utils import *

def main(input_path,noise_path,output_path, sampling_rate=16000):
    print(f"Input Path: {input_path}")
    print(f"Noise Path: {noise_path}")
    print (f"Output Path: {output_path}")
    config = {
        'speech_folder' : input_path,
        'noise_folder' : noise_path,
        'output_folder' : output_path,
        'sr' : sampling_rate,
        'snr_range': (5,15),
        'num_agumentations': 2
    }
    # Perform augmentation
    augment_dataset(
        config['speech_folder'],
        config['noise_folder'],
        config['output_folder'],
        sr=config['sr'],
        snr_range=config['snr_range'],
        num_augmentations=config['num_agumentations']
    )

    # Verify directory structure
    if verify_directory_structure(config['speech_folder'], config['output_folder']):
        print("Directory structure successfully preserved!")
    else:
        print("Warning: Directory structure might not match perfectly!")
        
    # Print summary
    create_dataset_summary(config['output_folder'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an input file and save the result to an output file.")
    
    # Define arguments
    parser.add_argument('--input', required=True, help="Path to the data-set folder")
    parser.add_argument('--noise', required=True, help="Path to the noise folder.")
    parser.add_argument('--output', required=True, help="Path to the augmented folder")
    parser.add_argument('--sampling', help="Sampling Rate")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with the arguments
    main(args.input,args.noise,args.output,args.sampling)
