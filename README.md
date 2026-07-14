# LungsAI
End-to-End Deep Learning Model for Lung Cancer Detection 
Designed and developed LungsAI, an end-to-end AI-powered clinical decision support platform for lung cancer detection from CT scan images. 
The project integrates deep learning, explainable AI, and a full-stack web application to assist healthcare professionals with automated and transparent diagnosis.
The solution compares ResNet-50 and EfficientNet-B0 transfer learning models, with EfficientNet-B0 + Focal Loss achieving the best performance (73.97% test accuracy).
 To improve model transparency, Grad-CAM was integrated to visualize the regions influencing predictions.
The trained model was deployed through a Flask REST API and integrated with a Next.js frontend, enabling secure clinician authentication, patient management, CT image upload, real-time AI prediction, confidence scoring, risk assessment, and prediction history.

Key Features:

End-to-end lung cancer classification using CT scan images

Transfer Learning with EfficientNet-B0 and ResNet-50

Explainable AI using Grad-CAM

Flask REST API for real-time inference

Next.js & Tailwind CSS clinical web application

Prisma ORM with SQLite/PostgreSQL

database integration

Patient management and prediction history
Image validation, confidence scoring, and clinical risk assessment

Technologies:
Python, PyTorch, EfficientNet-B0, ResNet-50, Flask, Next.js, TypeScript, Tailwind CSS, Prisma ORM, SQLite, PostgreSQL, REST API, Grad-CAM, NumPy, PIL, Git, GitHub.
