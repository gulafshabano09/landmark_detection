# landmark_detection
to make  a machine learning project on landmark detection with Python.

Have you ever looked through your vacation photos and wondered: what is the name of this temple I visited in India? Who created this monument that I saw in California? Landmark Detection can help us detect the names of these places. But how does the detection of landmarks work? I will introduce you to a machine learning project on landmark detection with Python.

## What is Landmark Detection?
Landmark Detection is a task of detecting popular man-made sculptures, structures, and monuments within an image. We already have a very famous application for such tasks which is popularly known as the Google Landmark Detection, which is used by Google Maps.

At the end of this article, you will learn how google landmark detection works as I will take you through a machine learning project which is based on the functionality of Google Landmark Detection. I will use the Python programming language for building neural networks to detect landmarks within images.

Now let’s get started with the task of detecting landmarks within an image. The most challenging task in this project is to find a dataset that includes some images that we can use to train our neural network. 

## Google Landmark Detection with Machine Learning
**Now to get started with this task, I will import all the necessary python libraries that we need to create a Machine Learning model for the task of landmark detection:**
![code1](https://user-images.githubusercontent.com/95492893/144703618-febda68b-578f-4806-a4fa-849ce3f7cfd0.PNG)

**So after importing the above libraries the next step in this task is to import the dataset that I will use for detecting landmarks withing images:**
![code2](https://user-images.githubusercontent.com/95492893/144703670-5652e61c-4acf-4760-8f10-16e02142221a.PNG)

![144703670-5652e61c-4acf-4760-8f10-16e02142221a](https://user-images.githubusercontent.com/95492893/144703683-10cb13b4-ae2a-46e3-9df5-c017bfbaee01.png)

**Now let’s have a look at the size of the training data and the number of unique classes in the training data:**
![code3](https://user-images.githubusercontent.com/95492893/144703750-3ef6d6fa-d0e2-4aac-bb36-b95ddb833e59.PNG)

![cod4](https://user-images.githubusercontent.com/95492893/144703780-4f1c8495-2a7e-4555-b14d-0edf0d16ddd6.PNG)

**There are 20,001 training samples, belonging to 1,020 classes, which gives us an average of 19.6 images per class, however, this distribution might not be the case, so let’s look at the distribution of samples by class:**
![code5](https://user-images.githubusercontent.com/95492893/144703838-dbebb273-d37f-4313-b156-6b817cbec9a9.PNG)

![code6](https://user-images.githubusercontent.com/95492893/144703904-a5ad30b3-7e6f-478e-b428-396ee295974d.PNG)

**As we can see, the 10 most frequent landmarks range from 139 data points to 944 data points while the last 10 all have 2 data points.**

![code7](https://user-images.githubusercontent.com/95492893/144703961-8d71abc2-8ce4-466a-bb28-f539930d76e2.PNG)

![code8](https://user-images.githubusercontent.com/95492893/144703985-4b2e11a9-a3b4-4625-9f63-0e22807bfded.PNG)

![image](https://user-images.githubusercontent.com/95492893/144703989-a0f09ad5-90e9-440d-9dc9-f0d5aa65bb22.png)

**As we can see in the histogram above, the vast majority of classes are not associated with so many images.**

![code9](https://user-images.githubusercontent.com/95492893/144704091-0a318e0b-b62f-4b69-8029-a0b061d8775f.PNG)

![code10](https://user-images.githubusercontent.com/95492893/144704120-f85b97be-6855-445f-9d95-bbc13c4c75be.PNG)

![image](https://user-images.githubusercontent.com/95492893/144704130-e2f47ea0-2007-4b71-9d52-9c2bf781b5b1.png)

**The graph above shows that over 50% of the 1020 classes have less than 10 images, which can be difficult when training a classifier.**
**There are some “outliers” in terms of the number of images they have, which means we might be biased towards those, as there might have a higher chance of getting a correct “guess” with the highest amount in these classes.**

## Training Model:##

**Now, I will train the Machine Learning model for the task of landmark detection using the Python programming language which will work the same as the Google landmark detection model.**
![code11](https://user-images.githubusercontent.com/95492893/144704253-66d057c1-0e36-4972-b0f7-0f4b0fc97a6a.PNG)
![code12](https://user-images.githubusercontent.com/95492893/144704287-5f5518d6-1032-447c-bb8c-2501f482beba.PNG)

![image](https://user-images.githubusercontent.com/95492893/144704323-b1646d7b-5872-42d3-ba7a-ec774caf1b3a.png)

![code13](https://user-images.githubusercontent.com/95492893/144704401-8b60f6d4-246d-43f5-a1b7-b111c88f65ef.PNG)
![code14](https://user-images.githubusercontent.com/95492893/144704425-6b630e28-4480-4940-862f-f6096f60de88.PNG)

![image](https://user-images.githubusercontent.com/95492893/144704525-62d7620b-a3a6-47c2-b153-a4d9d982c938.png)
![code16](https://user-images.githubusercontent.com/95492893/144704585-2d3af0eb-d90a-427d-ba2d-9449b12f65a7.PNG)
![code17](https://user-images.githubusercontent.com/95492893/144704620-5b0d1951-26b0-4daf-83b3-41b8a9165cec.PNG)

![code18](https://user-images.githubusercontent.com/95492893/144704654-34b84428-7003-41e8-8392-30a14ff097f7.PNG)
![code19](https://user-images.githubusercontent.com/95492893/144704680-33909215-4b70-473b-9aa7-5a95f66ea67d.PNG)

![code20](https://user-images.githubusercontent.com/95492893/144704695-490cf5f7-8b1b-4188-8790-5b5008947f0f.PNG)

**Now we have trained the model successfully. The next step is to test the model, let’s see how we can test our landmark detection model:**

![image](https://user-images.githubusercontent.com/95492893/144704735-33a224ee-1713-4ca5-bd19-2ddea7102d8d.png)
![code22](https://user-images.githubusercontent.com/95492893/144704758-ee8f8321-5ef3-4e13-9e0d-d407a76068f6.PNG)

![image](https://user-images.githubusercontent.com/95492893/144704770-7801cfb1-dc43-454f-9b36-fd8f3c885b18.png)

As we can see in the above images in the output, they are being classified according to their labels and classes.














































































