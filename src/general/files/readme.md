# Load the Train / Evaluation Files

Simple File Loader (Static) = file_manager

Useful Complete File Loaders = loaders

# Using the Loaders

Based on the ImageGenerator from Keras.io:

```python
    # Create a image loader object
    # The flows will get files from the subdirectories of the
    # path
    loader = ImageLoader(
        'C:/Training Images',
        size=(32, 32),
        # Default is gray
        channels = 'gray'
    )

    # Cropped From CSV
    flow = loader.crop_flow_from_csv('annotations.csv')

    # Common from CSV
    flow = loader.flow_from_csv('annotations.csv')

    # From Directory
    flow = loader.flow_from_directory(
        ['doors', 'indoors', 'stairs']
    )

    # Otherwise you can use a Feature Loader
    # It'll extract the features from the images
    # This is the default feature extractor
    feature_extract = lambda x: hog(x, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), feature_vector=True)
    # You can use others if you want ...
    featureLoader = FeatureLoader(
        'C:/Training Images',
        size=(32, 32)
    )
    featureFlow = flow = loader.flow_from_directory(['doors', 'indoors', 'stairs'])

    # Train the model with the Flow
    model.train_flow(flow)

    # Or using the FeatureLoader Flow
    model.train_flow(featureFlow)

    # You can also use this on a ImageGenerator from Keras.io
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    model.fit_generator(datagen.flow(flow.x(), flow.y(), batch_size=32),
                    steps_per_epoch=len(flow.x()), epochs=epochs)

```

A CSV for the Flow must contain relative sub directories e.g.:
class/image1.jpg, label

A Cropped CSV Flow must contain the directory and the labels for left, center, and right e.g.:
class/image1.jpg, class1, class2, class1