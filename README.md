# Team_Repo
# Food Delivery Prediction System

## Team members:
1. Satvir Kaur Mehra
2. Varinder Kaur
   
## Project Overview

This project aims to develop a machine learning system that predicts potential delays in food hamper deliveries, helping optimize logistics and improve the efficiency of food bank operations. The system allows for better planning, proactive identification of delays, and efficient resource allocation.

## Problem Statement

The delivery of food hampers faces challenges due to unforeseen delays. These delays can impact the timely distribution of food to those in need. By predicting delays in advance, logistics coordinators can take proactive steps to ensure on-time deliveries and reduce operational inefficiencies.

## Objectives

- Predict delays in food hamper deliveries.
- Improve delivery scheduling and reduce delays.
- Enhance communication between food banks and clients.
- Optimize delivery routes for efficient service.

## Key Features

- **Severity of Delay**: The severity of delay is predicted using machine learning models that assess the difference between actual delivery times and scheduled collection times. 
  - The wait time before observing delays is determined by comparing the actual delivery time (`datetime_from/datetime_to`) with the scheduled collection date (`collect_scheduled_date`).

- **Predictions and Value to End-User**:
  1. The model predicts delays in advance, allowing logistics teams to plan more effectively.
  2. The system provides alerts to delivery teams for high-risk delays, allowing them to prioritize critical deliveries.

- **Parameters in the Process**:
  - **Delivery Route Optimization**: Optimizing the routes to ensure faster and more efficient deliveries.
  - **Prioritization of At-Risk Deliveries**: Identifying deliveries that are most likely to be delayed, and prioritizing them.
  - **Automated Alerts**: Service providers are notified of deliveries with a high risk of delays.

## End-User and Objectives

### End-User

The end-users of the system are:
- **Food Bank Staff**
- **Logistics Coordinators**

These users rely on the system to:
- Identify potential delays in advance.
- Optimize delivery routes.
- Improve overall scheduling efficiency.

### Objectives of the End-User
- **Reduce Delays**: Minimize the occurrence of delayed deliveries.
- **Improve Resource Allocation and Planning**: Plan and allocate resources efficiently to handle delays and prevent bottlenecks.
- **Enhance Communication with Clients**: Keep clients informed about delivery times and any potential delays.

## How the ML System Benefits the End-User

- **Improved Efficiency in Delivery Scheduling**: The system helps logistics teams better plan delivery schedules by predicting potential delays.
- **Identification of Delay-Prone Areas or Clients**: The system can identify areas or clients that experience frequent delays, helping logistics coordinators improve service to these locations.
- **Proactive Resolution of Potential Delays**: With early alerts, delivery teams can act in advance to prevent or mitigate delays.

## Workflow and Interfaces

The system provides the following key interfaces:
- **Dashboard**: A real-time monitoring dashboard that displays predicted delays.
- **Automated Notifications**: Notifications for high-risk deliveries are automatically sent to delivery teams to ensure timely action.

## Conclusion

This project leverages machine learning to enhance the efficiency of food hamper deliveries. By predicting delays, optimizing delivery routes, and providing automated alerts, food banks and logistics coordinators can improve service, reduce delays, and better serve their communities.

## Link to Deployed App
https://teamrepo-8im6gcwbbqminyg2mwnrro.streamlit.app/
