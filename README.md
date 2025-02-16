
This project focuses on deploying an advanced mini rover to explore both unexplored and inhabited areas, gathering real-time environmental data. Equipped with object detection, motion detection, environmental analysis, and high-resolution imaging, the rover enables comprehensive monitoring of diverse environments.
By leveraging machine learning algorithms, the system processes and analyzes collected data autonomously, ensuring accurate insights for various applications. This enhances scientific research, disaster response, and environmental conservation by providing real-time, reliable information.It involves real tiem data collection,object and motion detection, image capturing and environmental analysis.

The robot system utilizes the YOLOv5 Nano model from Ultralytics, renowned for its real
time object detection capabilities, combining speed and accuracy. Its lightweight architecture allows for rapid inference across various hardware platforms, including constrained devices like the Raspberry Pi. 


<img width="561" alt="image" src="https://github.com/user-attachments/assets/7cf4aa65-1c79-4e22-ba1d-f51cf6e552b3" />



**Flask Integration -**

To facilitate seamless monitoring and control, the Raspberry Pi is configured to function as a web server. This setup enables operators to access and manage the robot's activities through a user-friendly ebpage interface. By connecting to the web server, users can receive real-time updates, view live data feeds, and make informed decisions based on the comprehensive information gathered by the robot. 
 
The primary objective of deploying this robotic system is to achieve a holistic understanding of the target area. This understanding is crucial for making well-informed decisions and implementing necessary interventions, whether for scientific research, safety assessments, or operational planning. Through the continuous real-time data provided by the robot, operators can effectively monitor and respond to the dynamic conditions of the environment, ensuring optimal outcomes for the mission at hand.


<img width="561" alt="image" src="https://github.com/user-attachments/assets/e0d3396a-61bb-4fdf-b08b-0dd02cdbd0cf" />


This project utilizes TCP and HTTP protocols to establish seamless communication between the Raspberry Pi and external systems, enabling remote access, control, and web-based interactions.

**TCP Protocol for Remote Access**
To establish a reliable connection between the Raspberry Pi and a local machine, the RealVNC API is employed. This allows for remote access to the Raspberry Pi’s GUI, enabling users to execute commands, run applications, and perform maintenance tasks efficiently.

**HTTP Protocol for Web-Based Interaction**
The Flask framework is used to host a webpage, leveraging the Hypertext Transfer Protocol (HTTP) for communication. Flask simplifies request handling using decorators like @app.route, mapping URLs to specific functions for seamless web interactions.
