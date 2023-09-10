from ultralytics import YOLO
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Initialize parser
msg = "Trainer for yolov8 model"
parser = argparse.ArgumentParser(description = msg)

#add model argument and read
parser.add_argument("-m", "--model", help = "Path of model to train. Uses yolov8m if not passed.")
parser.add_argument("-r", "--resume", help = "Resumes progress. true/false.", action='store_true')
parser.add_argument("-e", "--epochs", help = "Number of epochs")
parser.add_argument("-b", "--batch_size", help = "Batch size")
args = parser.parse_args()



if __name__ == '__main__':
    if args.model:
        modelName = args.model
        print("Training model: % s" % args.model)
    else:
        print("Using default: yolov8s.pt")
        modelName = 'yolov8s.pt'
    epochNum = args.epochs if args.epochs else 100
    batchSize = args.batch_size if args.batch_size else 10
    print("Resuming: " + str(args.resume))
    print("Batch Size: " + str(batchSize))
    print("Epochs: " + str(epochNum))
    # Load a model
    model = YOLO(modelName)  # load a pretrained model (recommended for training)
    # Train the model
    model.train(data='annotations/data.yaml', epochs=int(epochNum), batch=int(batchSize), device=0, resume=args.resume)