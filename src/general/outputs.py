def show_label(label, item):
    imcls = 'door' if item == 1 else 'indoor'
    return label + ' - '+ imcls

def show_images_predictions(left, center, right):
    import matplotlib.pyplot as plt
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                    ncols=3,
                                    sharex=True,
                                    sharey=True)
    left_label = show_label('left', left[1])
    ax0.imshow(left[0], cmap='gray')
    ax0.set_title(left_label)
    ax0.axis('off')
    ax0.set_adjustable('box-forced')

    center_label = show_label('center', center[1])
    ax1.imshow(center[0], cmap='gray')
    ax1.set_title(center_label)
    ax1.axis('off')
    ax1.set_adjustable('box-forced')

    right_label = show_label('right', right[1])
    ax2.imshow(right[0], cmap='gray')
    ax2.set_title(right_label)
    ax2.axis('off')
    ax2.set_adjustable('box-forced')

    fig.tight_layout()

    return fig

def show_detections(detections, image):
    import cv2
    # Display the results after performing NMS
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(image, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
    cv2.imshow("Final Detections after applying NMS", image)
    cv2.waitKey()

def visualize_sliding(image, im_window, pos, cd):
    import cv2
    clone = image.copy()
    for x1, y1, _, _, _  in cd:
        # Draw the detections at this scale
        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
            im_window.shape[0]), (0, 0, 0), thickness=2)
    cv2.rectangle(clone, pos, (pos[0] + im_window.shape[1], pos[1] +
        im_window.shape[0]), (255, 255, 255), thickness=2)
    cv2.imshow("Sliding Window in Progress", clone)
    cv2.waitKey(30)