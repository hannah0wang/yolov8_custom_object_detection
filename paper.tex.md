\documentclass[conference,compsoc]{IEEEtran}
\usepackage{graphicx} % Required for inserting images
\usepackage[utf8]{inputenc}
\usepackage{lipsum}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{indentfirst}

\title{Custom Overpass Detection on Roads and Highways Using YOLOv8}

\date{November 2023}

\begin{document}

\maketitle

\begin{abstract} 
This study focuses on YOLOv8, a state-of-the-art object detection model, and aims to optimize its overpass detection capabilities. Through the analysis of different performance metrics and dataset improvements via augmentations, the study aims to improve detection precision on a custom dataset of overpasses. Upon demonstrating relative accuracy, the model underwent testing on dashcam footage synchronized with data such as GPS coordinates, time, and date. This comprehensive study not only showcases the model's capabilities in real-world scenarios but also highlights the potential for extrapolating critical ground truth from overpass detections, particularly the exact vehicle location and timing while passing overpasses.\\   
\end{abstract}

\providecommand{\keywords}[1]
{
  \small	
  \textbf{\textit{Keywords---}} #1
}
\keywords{Overpass Detection, YOLOv8, Performance Metrics}

\section{Introduction}
\subsection{Motivation}
Advancements in computer vision methodologies have significantly enhanced the accuracy and efficiency of object identification. 
\href{https://www.sciencedirect.com/science/article/pii/S2666827021000670}{(Chai, Junyi, et al.)}. These advancements serve as compelling reasons to extend object detection capabilities to encompass more complex objects and dynamic points of reference in motion, especially in the context of roads and highways. In this study, detecting overpasses holds a pivotal role in establishing ground truth regarding a vehicle's location and timestamp when passing under or near an overpass within a transportation network. Pinpointing the exact location and time when a vehicle traverses under an overpass assists in precise geographical mapping and creates an accurate records of a vehicle's journey.

\subsection{ YOLOv8}
YOLOv8 is the latest evolution of the You Only Look Once (YOLO) model, released in 2023. It was developed by Ultralytics and represents a significant advancement in object detection algorithms within the realm of computer vision. YOLO is renowned for its real-time object detection capabilities, and the latest version aims to enhance precision, speed, and versatility in identifying objects within images or video frames.  \href{https://docs.ultralytics.com/}{(Ultralytics)}
The algorithm operates by dividing the input image into a grid and performing object detection and classification for each grid cell, thereby enabling swift and accurate localization of multiple objects simultaneously. YOLOv8 can be run from the command line interface (CLI) or installed as a PIP package. For this study, the CLI was used for training and inference tasks via Google Colaboratory, a Jupyter Notebook-like environment with access to Tesla K80 GPU
\href{https://colab.research.google.com/}{(Google Colab)}.
YOLOv8 provides five scaled versions: YOLOv8n (nano), YOLOv8s (small), YOLOv8m (medium), YOLOv8l (large) and YOLOv8x (extra-large)
\href{https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes}{Ultralytics}.
Smaller models are lighter weight but sacrifice some accuracy for faster inference speed. YOLOv8m was used in this study as it balances the trade-off between accuracy and inference speed.

\subsection{Object Detection Metrics and Bounding Boxes}
Average Precision (AP) is a widely used evaluation metric in the field of object detection, particularly in assessing the performance of models that identify and localize objects within images or videos
\href{https://arxiv.org/pdf/2304.00501v1.pdf}{(Terven, Juan R., and Diana M. Cordova-Esparaza)}.
The AP metric utilizes the Intersection over Union (IoU) measure to assess the quality of predicted bounding boxes. IoU is defined as the ratio of the area of overlap between the predicted bounding box and the ground truth bounding box to the area of their union
\href{https://giou.stanford.edu/}{(Rezatofighi, Hamid, et al.)}.
The formula for IoU is given below:\\
\[ IoU = 
\frac{Area of Intersection}{Area of Union} = 
\frac{X \cap Y}{X \cup Y}
\]
Where X represents the ground truth and Y represents the predicted bounding box.
\begin{figure}[htp]
    \centering
    \includegraphics[width=8cm]{new-iou.png}
    \caption{Intersection over Union. a) The IoU is calculated by dividing the intersection of the two boxes by the union of their boxes; b) examples of three different IoU values for different box locations. Figure sourced from Terven, Juan R., and Diana M. Cordova-Esparaza.}
    \label{fig:iou}
\end{figure}
\subsection{Confidence Score as a Performance Indicator}
The confidence score of a bounding box is a formal measure calculated by comparing the predicted bounding box with the corresponding ground truth box, known as Intersection over Union (IoU)
\href{https://www.sciencedirect.com/science/article/pii/B9780323857871000166}{(Iosifidis, Alexandros, et al.)}.
This score reflects the model's certainty regarding the presence of a detected object within the predicted bounding box. Analyzing these confidence scores enables the establishment of thresholds to accept or reject instance detections. These thresholds can be adjusted using experimental data to minimize false positives and improve the model's overall accuracy. In this particular study, a generous confidence threshold of 0.5 was chosen to ensure capturing all validated instances. Given the fixed nature of overpasses, their precise GPS coordinates allow verification, enabling the dismissal of false positives if their predicted locations do not align with any known overpass locations.

\begin{figure}[htp]
    \centering
    \includegraphics[width=3cm]{person-conf.png}
    \caption{Example bounding box with confidence score \href{Detect - Ultralytics YOLOv8 Docs}{Detect-Ultralytics}}
    \label{fig:person}
\end{figure}

\subsection{The Need for Training YOLOv8 Object Detection on a Custom Dataset}
YOLOv8 uses the Microsoft COCO (Common Objects in Context) dataset as the standard benchmark for evaluating performance and includes 91 object categories with a total of 2.5 million labeled instances in 328k images

\href{https://www.microsoft.com/en-us/research/wp-content/uploads/2014/09/LinECCV14coco.pdf}{(Lin, Tsung-Yi, et al.)}. While the COCO dataset is diverse, it does not encompass every possible category that might be encountered in real-world applications. This study, centered on the in-depth analysis of overpass instances along roads and highways, an object not present in the COCO dataset, necessitated training the YOLOv8 model with a custom dataset comprising overpass images.

\section{Methodology}
\subsection{Collecting Images to Create the Overpass Dataset}
The dataset used in this custom model sourced images from 32 dashcam videos of individual drives synchronized with GPS, time, and date. Each instance of an overpass within a video was identified through a manual annotation process. This manual identification involved a visual inspection of the video sequences and identifying the specific time frames in which the overpass instances occurred. In the first trained and tested model, snapshot photos of each overpass were taken randomly to generate a dataset of 50 images containing overpass instances. Minor rotations were applied to the images, increasing the dataset to 67 images. After generating unsatisfactory results (refer to Section [insert section number]), we realized that the dataset was rather limited and needed to be expanded for more accurate detection
\href{https://link.springer.com/article/10.1007/s10278-021-00536-0}{(Sanaat, Amirhossein, et al.)}.
outlines the effects of small datasets and methods for improving performance and robustness in such scenario, including the application of augmentations to the limited dataset. In addition to applying augmentations (listed below), snapshot photos were captured in 1 second intervals instead of randomly from the time an overpass reaches reasonable view (overpass is identified based on human discretion) to the time it leaves the frame. This approach not only enabled the proliferation of the dataset by generating a substantial volume of images given the sporadic occurrence of overpasses, but also ensured the overpasses were captured from various angles and points of view. This contributes significantly to enhancing the accuracy and robustness of the model used for overpass detection. From this process, we were able to produce 94 images, almost doubling our initial dataset.

\subsection{Creating the Custom Dataset in Roboflow}
Roboflow is an end-to-end computer vision platform tailored for the comprehensive management, annotation, preprocessing, and augmentation of image datasets, specifically designed to support tasks in computer vision and machine learning. It encompasses an array of tools and functionalities aimed at streamlining the preparation and refinement of image datasets for training machine learning models
\href{https://roboflow.com}{(Roboflow)}.
The creation of the custom dataset involved the manual labeling and annotating of all 94 images within the raw dataset, employing Roboflow's bounding box annotation tools. While it was a tedious process to label all images by hand, the precise delineation and annotation of overpasses presented in the images establish comprehensive ground truth essential for subsequent model training and analysis.

\begin{figure}[htp]
    \centering
    \includegraphics[width=6.5cm]{ex-bounding-box.png}
    \caption{Example labeling on an image from the dataset}
    \label{fig:matrix}
\end{figure}

\subsection{Applying Augmentations to the Dataset}
Next, Roboflow’s data augmentation techniques were leveraged to improve model generalization and robustness as well as to increase the dataset volume. The 10 following augmentations were applied randomly (and occasionally in combination with others) to photos in the training dataset:

\begin{itemize}
    \item Cropping, 0\% minimum zoom and 42\% maximum zoom to help the model be more resilient to subject translations and camera position.
    \item Rotations between -41 and 41 degrees to help model detect objects even when the camera or subject are not perfectly aligned.
    \item Grayscale applied to 25\% of the images to increase training variance.
    \item Exposure between -20\% and 20\% to help model be more resilient to lighting and camera setting changes.
    \item Random Gaussian blur up to 2.5px to help model be more resilient to poor camera focus or to mitigate the effects of overfitting on hard edges.
    \item Noise up to 1\% of pixels to help model be more resilient to camera artifacts (precipitation, dust, etcetera).
    \item Cutouts, 3 boxes with 10\% size each to help model be more resilient to object occlusion (other vehicles or objects in environment blocking full view).
    \item Mosaic (combination of training set images in a collage).
    \item Bounding box shear to help model be more resilient to camera and subject pitch and yaw.
    \item Random Gaussian blur on bounding box to improve resilience against poor camera focus.
\end{itemize}

\begin{figure}[htp]
    \centering
    \includegraphics[width=7cm]{roboflow-augmentations.png}
    \caption{Example training dataset photos with augmentations applied to improve model accuracy.}
    \label{fig:augment}
\end{figure}

A total of 226 images were generated by applying augmentations to the original 94 images, reflecting a 237\% increase in the dataset size compared to the original dataset of only 67 images. The images in the new dataset were distributed across different sets: 88\% (198 images) in the training set, 8\% (19 images) in the validation set, and 4\% (9 images) in the test set. These proportions were recommended by Roboflow to optimize training efficiency and model accuracy.
In this setup, the training set serves as the subset from which the model learns to make predictions. The validation set plays a crucial role in hyperparameter tuning, facilitating the evaluation of augmentation techniques and aiding in the selection of optimal hyperparameters by assessing the model's performance on this specific subset. The test set is reserved to provide an impartial and unbiased estimate of the final model's performance
\href{https://www.statology.org/validation-set-vs-test-set/#:~:text=Whenever%20we%20fit%20a%20machine%20learning%20algorithm%20to,an%20unbiased%20estimate%20of%20the%20final%20model%20performance.}{(Zach)}.

\begin{figure}[htp]
    \centering
    \includegraphics[width=7.5cm]{set-splits.png}
    \caption{Example dataset split sourced from Statology (Zach).}
    \label{fig:split}
\end{figure}

After creating the dataset, it was subsequently exported from Roboflow and into Google Colab to facilitate the training and testing processes.


\section{Experimental Results}
\subsection{Video Observation}
After the initial dataset of 67 images was created, the model underwent training using this dataset and was subsequently executed on a dashcam video for evaluation. As observed in the figures below, the model exhibited significant inaccuracy, consistently detecting overpasses even when no instances existed in the current frame. Despite the model successfully detecting all instances of overpasses (which are few in reality), its functionality proved futile as it continuously misidentified other objects as overpasses.

\begin{figure}[htp]
    \centering
    \includegraphics[width=9.5cm]{old-model-video.png}
    \caption{Example frames from the dashcam video after running the detection model.}
    \label{fig:oldvid}
\end{figure}

Illustrated in Figure 6, instances were detected frequently, often in substantial numbers, across nearly every frame of the video. However, it's important to note that the presence of overpasses was confined to only a few frames. This excessive overprediction by the model renders its outcomes inconclusive, given the disparity between the actual instances of overpasses and the model's detections.The unsatisfactory performance can be attributed to the model's inadequate learning opportunities due to the limited dataset.

\begin{figure}[htp]
    \centering
    \includegraphics[width=9.5cm]{new-model-video.png}
    \caption{Left: Model correctly detecting an overpass instance. Right: Model correctly identifying no overpasses.}
    \label{fig:newvid}
\end{figure}

Following the implementation of the dataset expansion techniques outlined in Section 2 and subsequently running the model on the same dashcam video, a significant improvement in detection accuracy was observed. The model notably reduced its incidence of false positives, which is reflected in Figure 7.

\subsection{Confusion Matrix to Analyze Model Performance}
The assessment of model detections involves comparing them against ground truth and categorizing them into four groups. Firstly, True Positives (TP) occur when the model accurately identifies an object. False Positives (FP) happen when the model detects an object that is not actually present. False Negatives (FN) arise when an object in the ground truth remains undetected by the model. Lastly, True Negatives (TN) represent correctly undetected background objects, but they are typically not considered in object detection evaluations. Precision reflects how accurately the model predicts, while Recall measures its ability to predict correctly when required. For instance, consider a scenario with two overpasses in an image, where the model correctly identifies only one. In this case, the model demonstrates perfect precision (all its guesses are accurate), but imperfect recall (only one out of the two overpasses is detected)
\href{https://blog.roboflow.com/mean-average-precision/#frequently-asked-questions}{(Solawetz, Jacob)}.

\begin{figure}[htp]
    \centering
    \includegraphics[width=8cm]{roboflow-conf-matrix.png}
    \caption{Example confusion matrix graphic with calculations for precision and recall metrics. Graphic sourced from Roboflow.}
    \label{fig:robomatrix}
\end{figure}

\begin{figure}[htp]
    \centering
    \includegraphics[width=6.5cm]{old-confusion-matrix.png}
    \caption{The resulting confusion matrix after training the model on the initial custom overpass dataset (validation set)}
    \label{fig:oldmatrix}
\end{figure}

The confusion matrix in Figure 9 comprising of data from the first trained model reveals that only 7 overpasses were accurately identified (TP). There were 60 instances where no detection was anticipated but detection occurred (FP). Zero overpasses went undetected when it was expected to be identified (FN).\\

The Precision and Recall for the initial dataset can be calculated as: \\
\[ Precision = 
\frac{TP}{TP + FP} = 
\frac{7}{7 + 60} = 10.44\%
\]
\[ Recall = 
\frac{TP}{TP + FN} = 
\frac{7}{7 + 0} = 100\%
\] \\

The Recall is 100\% due to nearly each frame containing a detection, so the model is bound to correctly identify when there actually is an overpass instance. However, with a Precision of only 10.44\%, the model demonstrates considerable inaccuracy in its detections due to a high number of false positives.\\


\begin{figure}[htp]
    \centering
    \includegraphics[width=6cm]{new-confusion-matrix.png}
    \caption{The resulting confusion matrix after training the model on the improved custom overpass dataset (validation set) }
    \label{fig:newmatrix}
\end{figure}


The confusion matrix in Figure 10 reveals that out of the overpasses present, 10 were accurately identified (TP), 6 were incorrectly detected when no detection was anticipated (FP), and 1 overpass went undetected when it was expected to be identified (FN).

The Precision and Recall for the initial dataset can be calculated as: \\
\[ Precision = 
\frac{TP}{TP + FP} = 
\frac{10}{10 + 6} = 62.5\%
\]
\[ Recall = 
\frac{TP}{TP + FN} = 
\frac{10}{10 + 1} = 90.9\%
\] \\

The decrease in Recall to 90.9\% might be attributed to the expanded dataset, which now includes overpasses captured from less optimal angles. However, this change coincided with a significant increase in Precision.

\begin{figure}[htp]
    \centering
    \includegraphics[width=8.5cm]{undetected-overpasses.png}
    \caption{Instances where overpasses went undetected in the improved model.}
    \label{fig:undetect}
\end{figure}

These instances in Figure 11 represent false negatives, suggesting that augmentations applied to the training set could be intensified. Implementing more aggressive rotations, shear adjustments, and larger cutouts might further mitigate inaccuracies.

\subsection{Loss Analysis}
Loss functions play a crucial role in assessing a model's performance with its provided data and its capacity to forecast anticipated results. These functions are integral to numerous machine learning algorithms, aiding in the optimization process during training to refine output accuracy
\href{https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9}{(Yathish, Vishal)}.

By defining objectives against which a model's performance is measured, loss functions guide the model's learning process, determining optimal parameters through the minimization of the chosen loss function. Throughout the training phase for each dataset in this study, the model computes three distinct losses: box loss, classification loss, and objectness loss. Box loss quantifies the algorithm's proficiency in locating an object's center and accurately encompassing it within the predicted bounding box. Classification loss measures the algorithm's capability to predict the correct class for a given object. And lastly, objectness reflects the likelihood that an object resides within a suggested region of interest; higher objectness signifies a higher probability of an object's presence within that image window
\href{https://www.researchgate.net/publication/350540629_Short_Communication_Detecting_Heavy_Goods_Vehicles_in_Rest_Areas_in_Winter_Conditions_Using_YOLOv5}{(Berger, Stian, et al.)}.
Ideally, as the number of epochs (times the entire dataset is presented to the model) increases, each loss function should eventually approach zero, indicating improved model convergence and enhanced accuracy in prediction.\\

\begin{figure}[htp]
    \centering
    \includegraphics[width=8.5cm]{old-loss.png}
    \caption{Plots of box loss, classification loss, and objectness loss for the initial dataset.}
    \label{fig:oldloss}
\end{figure}

\begin{figure}[htp]
    \centering
    \includegraphics[width=8.5cm]{new-loss.png}
    \caption{Plots of box loss, classification loss, and objectness loss for the improved dataset.}
    \label{fig:newloss}
\end{figure}

As observed in Figure 12, the loss functions increased after the training period, indicating that the model's predictions are progressively diverging from the actual ground truth. The model’s failure to converge is highlighted when it was employed to detect overpasses within the dashcam video, as the model was unable to distinguish between the overpasses and the surrounding background. On the other hand, Figure 13 reveals a convergence trend across all the loss functions, suggesting that the augmentations employed to expand the dataset proved effective in enhancing the model's performance. This improvement was evident in the improved performance of overpass detection within the dashcam video.

\subsection{Extracting Ground Truth from Video Frame Predictions}
The model's reasonably accurate predictions present the opportunity for uncovering further insights. During model detection on images or videos, real-time outputs display detections for objects within each frame of a video stream. This output includes information about whether an overpass is present in a certain frame of the video and the amount of time taken to produce a prediction.

\begin{figure}[htp]
    \centering
    \includegraphics[width=9.5cm]{stream-output-small.png}
    \caption{A segment of the output generated during the model's object detection process.}
    \label{fig:stream}
\end{figure}

In Figure 15, the first column to the left represents the class number (zero indexed), with instances solely denoted as zero since the dataset exclusively comprises overpasses. Columns 2 and 3 display the x and y coordinates of the center of the bounding box. The last value is the confidence score. Columns 4 and 5 indicate the width and height of the bounding box, starting from its center coordinates. The last column provides the confidence score associated with each detection. Only detections surpassing a confidence score of 0.5 are recorded in this output file, since the confidence threshold was set to 0.5.

\begin{figure}[htp]
    \centering
    \includegraphics[width=8cm]{save-conf-true.png}
    \caption{A segment of the outputs saved to text file.}
    \label{fig:saveconf}
\end{figure}

\begin{figure}[htp]
    \centering
    \includegraphics[width=8cm]{combined-txt-file.png}
    \caption{An excerpt from the combined text file provides timestamp associations for each prediction.}
    \label{fig:combtxt}
\end{figure}

By employing ffmpeg to extract frame rate information from the video and utilizing Python for text file manipulation, it enables each prediction to be associated with a timestamp in the video, as shown in Figure 16.

These timestamps represent the time in the specific video that an overpass prediction occurs. Following this, we manually cross-referenced the predictions with the actual videos, confirming the model's successful detection of every overpass instance. However, there are a couple instances of false positives detected throughout the video. But these detections are usually brief and span only a couple of frames compared to when true positives occur.\\\\
Since we have data about the exact time and date that each drive in the dashcam video occurred, we can utilize the timestamps from overpass predictions within the video to estimate the exact time and date of the vehicle passing an overpass. Furthermore, as each dashcam video is time-synchronized with GPS data, we can verify the vehicle's precise location by comparing the GPS prediction of each overpass within the video to its actual location. However, due to time constraints, further pursuits involving the verification of overpass locations using additional methods, despite the available synchronized GPS data and dashcam videos, were not undertaken within the scope of this study.

\section{Conclusion}
Section 3 highlights a notable enhancement in the model's overpass detection capability; however, opportunities for improvement persist by enlarging the raw dataset and implementing additional augmentations. A comparison of YOLOv8's performance on the COCO dataset, composed of hundreds of thousands of images, underscores the potential for further refinement in the model's proficiency. In an ideal scenario, expanding the overpass dataset to match the scale of the COCO dataset could potentially lead to a comparable level of accuracy and robustness. However, for the purpose of this study, the model's performance was proficient enough to allow the extraction of relevant ground truth. Being able to derive time predictions for each instance of overpass detection presents an opportunity to extrapolate relative GPS coordinates of the vehicle. As overpasses are typically stationary structures, their consistent locations offer a reference point for gauging the vehicle's position at a specific time. Utilizing this data could potentially provide insights into the vehicle’s orientation and its direction of travel.

\section{Acknowledgements}
We would like to thank Alex Richardson and Kate Sanborn for their prior efforts in aligning dashcam footage with relevant car data, establishing the essential foundation for our present analysis.

\section{References}

Berger, Stian, et al. “Short communication: Detecting heavy goods vehicles in rest areas in winter conditions using yolov5.” Algorithms, vol. 14, no. 4, 2021, p. 114, https://doi.org/10.3390/a14040114.\\\\
Chai, Junyi, et al. “Deep Learning in Computer Vision: A critical review of emerging techniques and application scenarios.” Machine Learning with Applications, vol. 6, 14 Aug. 2021, https://doi.org/10.1016/j.mlwa.2021.100134. \\\\
Google Colab, Google, colab.research.google.com/. Accessed 3 Dec. 2023. \\\\
Iosifidis, Alexandros, et al. “Chapter 11 - Object Detection and Tracking.” Deep Learning for Robot Perception and Cognition, Academic Press, Amsterdam, 2022, pp. 243–278. \\\\
Lin, Tsung-Yi, et al. “Microsoft Coco: Common Objects in Context.” Computer Vision – ECCV 2014, 2014, pp. 740–755, https://doi.org/10.1007/978-3-319-10602-1\_48. \\\\
Rezatofighi, Hamid, et al. “Generalized intersection over union: A metric and a loss for bounding box regression.” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, https://doi.org/10.1109/cvpr.2019.00075. \\\\
Roboflow, roboflow.com/. Accessed 3 Dec. 2023. \\\\
Sanaat, Amirhossein, et al. “Robust-deep: A method for increasing brain imaging datasets to improve deep learning models’ performance and Robustness.” Journal of Digital Imaging, vol. 35, no. 3, 2022, pp. 469–481, https://doi.org/10.1007/s10278-021-00536-0. \\\\
Solawetz, Jacob. “Mean Average Precision (MAP) in Object Detection.” Roboflow Blog, Roboflow Blog, 25 Nov. 2022, blog.roboflow.com/mean-average-precision/\#frequently-asked-questions. \\\\
Terven, Juan R., and Diana M. Cordova-Esparaza. “A COMPREHENSIVE REVIEW OF YOLO: FROM YOLOV1 TO YOLOV8 AND BEYOND.” ACM COMPUTING SURVEYS, 4 Apr. 2023. \\\\
Ultralytics YOLOv8 Docs, docs.ultralytics.com/. Accessed 3 Dec. 2023. \\\\
Ultralytics. “Detect.” Detect - Ultralytics YOLOv8 Docs, docs.ultralytics.com/tasks/detect/. Accessed 3 Dec. 2023. \\\\
Ultralytics. Ultralytics YOLOv8 Docs, docs.ultralytics.com/models/yolov8/\#supported-tasks-and-modes. Accessed 3 Dec. 2023. \\\\
Yathish, Vishal. “Loss Functions and Their Use in Neural Networks.” Medium, Towards Data Science, 4 Aug. 2022, towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9. \\\\
Zach. “Validation Set vs. Test Set: What’s the Difference?” Statology, 20 Sept. 2021, www.statology.org/validation-set-vs-test-set/\#:~:text=Whenever\%20we\%20fit\%20a\%20machine\%20learning\\\%20algorithm\%20to,an\%20unbiased\%20estimate\%20of\%20the\\\%20final\%20model\%20performance. 

\end{document}
