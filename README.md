#  🛰️ Satellite-Image-Segmentation-using-U-Net
SatelliteSeg AI uses U-Net with a MobileNetV2 encoder to detect and segment buildings in satellite images. It provides image preprocessing, customizable detection thresholds, segmentation masks, visual overlays, building coverage statistics, pixel analysis, and downloadable results through an interactive Streamlit dashboard.

# 🛰️ SatelliteSeg AI

## 📌 Project Overview
SatelliteSeg AI is a deep learning application that detects and segments buildings in satellite images using U-Net architecture with a MobileNetV2 encoder. It provides an interactive Streamlit dashboard to upload satellite images, generate segmentation masks, visualize detected buildings, analyze pixel coverage, and download prediction results.

## ✨ Features
- 🛰️ Satellite image upload (JPG, JPEG, PNG)
- 🧠 U-Net-based building segmentation
- ⚙️ MobileNetV2 encoder
- 🖼️ Automatic image preprocessing and resizing
- 🎚️ Adjustable building detection threshold
- 🎭 Predicted segmentation mask visualization
- 🔍 Segmentation overlay on the original image
- 📊 Pixel analysis and building coverage statistics
- 💾 Download segmentation masks and overlay images
- 🎨 Interactive Streamlit dashboard

## 🧠 Model and Processing

- 🏗️ **Model Architecture:** U-Net
- ⚙️ **Encoder:** MobileNetV2
- 🏢 **Task:** Binary Building Segmentation
- 📐 **Input Image Size:** 256 × 256 pixels
- 🎭 **Output:** Binary Segmentation Mask
- 📉 **Loss Functions:** Binary Cross-Entropy and Dice Loss

## 🔄 Processing Pipeline

1. 📤 Upload a satellite image.
2. 🎨 Convert the image to RGB format.
3. 📐 Resize the image to 256 × 256 pixels.
4. 🔢 Normalize pixel values between 0 and 1.
5. 🧠 Generate predictions using the trained U-Net model.
6. 🎚️ Apply the selected threshold to create a binary segmentation mask.
7. 🔍 Generate an overlay highlighting predicted building regions.
8. 📊 Display segmentation results and building coverage statistics.
9. 💾 Download the generated mask and overlay images.

## 🛠️ Technologies Used
- 🐍 Python
- 🤖 TensorFlow
- 🧠 Deep Learning
- 🏗️ U-Net
- 📡 MobileNetV2
- 🔢 NumPy
- 🖼️ Pillow (PIL)
- 🌐 Streamlit
- 📊 Image Segmentation

  ## 🚀 Future Scope

- 🛰️ Improve building detection accuracy using advanced deep learning models.
- 🌍 Extend segmentation to roads, rivers, vegetation, and other satellite features.
- 📡 Integrate real-time satellite imagery and remote sensing data.
- 🏙️ Develop urban planning and infrastructure monitoring applications.
- 📊 Add advanced analytics for building density and land-use estimation.
- ☁️ Deploy the application on cloud platforms for scalable access.
- 📱 Develop a mobile-friendly interface for on-the-go analysis.
- 🧠 Explore advanced architectures such as Attention U-Net and Transformer-based segmentation.
- 🗺️ Integrate GIS tools for geographic visualization and mapping.
- 🔄 Support high-resolution satellite images and large-scale image processing.

## 🎯 Conclusion

SatelliteSeg AI demonstrates the application of deep learning in satellite image analysis using U-Net with a MobileNetV2 encoder. The project enables building segmentation, mask generation, visual overlay creation, and building coverage analysis through an interactive Streamlit dashboard. It provides a useful foundation for future applications in urban planning, infrastructure monitoring, and remote sensing.
