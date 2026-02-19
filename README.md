VIdhaiX - Your all-in-one 
--------------------------

Vidhaix is a smart farming assistant built to support farmers in making better informed decisions. It provides real-time mandi price insights, crop planning guidance, and simple predictive analytics to help farmers decide what to grow and when to sell. By combining live market trends with region-specific recommendations, the platform aims to reduce uncertainty in farming and help farmers improve their profits while practicing more sustainable agriculture. 

<img width="1600" height="708" alt="image" src="https://github.com/user-attachments/assets/ee4b9cb0-f218-4f5b-93fa-06f0820e9b64" />
<img width="1600" height="708" alt="image" src="https://github.com/user-attachments/assets/ee4b9cb0-f218-4f5b-93fa-06f0820e9b64" />



We have implemented Machine Learning to get the land details as input from the user and suggest the best dual or multi-cropping plan that can be implemented. This is done by feeding a database ‘Crop Recommendation for the past 10 years’ (that includes various conditions and the crops that thrive in those conditions) into a Random Forest Classifier model and training it so that it recommends the suitable crops that can be planted in the given area. 
<img width="1600" height="744" alt="image" src="https://github.com/user-attachments/assets/209381a5-3bb1-47b8-8dcc-27aad349c836" />
<img width="1600" height="744" alt="image" src="https://github.com/user-attachments/assets/209381a5-3bb1-47b8-8dcc-27aad349c836" />

It also considers factors such as seasonal compatibility, soil health sustainability, and risk reduction while proposing multi-cropping plans. These recommendations help farmers choose crop combinations that maximize yield, improve soil utilization, and ensure more stable and profitable farming outcomes.


The market price comparison is done by taking real-time data from government websites using API and storing it in a database. This is a hybrid model in which instead of using either only API or only pre-defined database, we use a combination of both. The system periodically collects mandi price data, stores it, and processes it to generate meaningful comparisons and trends. 
![WhatsApp Image 2026-02-19 at 6 11 32 PM](https://github.com/user-attachments/assets/7adf496d-aafe-4a28-a508-03915c4de296)
![WhatsApp Image 2026-02-19 at 6 11 32 PM](https://github.com/user-attachments/assets/7adf496d-aafe-4a28-a508-03915c4de296)
<img width="1522" height="710" alt="image" src="https://github.com/user-attachments/assets/d50d5b17-06f3-4076-8074-b6f66bed62f6" />
<img width="1522" height="710" alt="image" src="https://github.com/user-attachments/assets/d50d5b17-06f3-4076-8074-b6f66bed62f6" />


This allows farmers to view current prices, track historical changes, and understand daily market fluctuations through a simple dashboard. By analyzing both live and stored data, the module highlights price movements and better selling opportunities across markets. This helps farmers decide the right time and place to sell their produce with greater confidence.

Vidhaix ultimately strives to bridge the gap between what is grown in the field and what is demanded in the market, making technology a practical ally in everyday farming decisions.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Future Development Plans:

 In the future, we aim to develop Vidhaix into a full-fledged mobile application that can be easily accessed by farmers across different regions. We also plan to integrate more advanced AI techniques to improve the accuracy and reliability of crop recommendations and market predictions. To make the platform more inclusive, we intend to support voice-based interactions so that users can navigate and receive insights without needing to type. Additionally, the application will provide multilingual support, automatically adapting the content to regional languages, ensuring that even illiterate or physically challenged farmers can use the system comfortably and effectively.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Individial Contributions:

Armitha J:
 - entire market price module

Jayashree K J
 - weather and soil API

Mani Nila S
 - combining and intergrating the final website









Vidhaix ultimately strives to bridge the gap between what is grown in the field and what is demanded in the market, making technology a practical ally in everyday farming decisions.

