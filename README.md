# Silatus
Automated Landing Page Description Generator


### Introduction
In today's fast-paced digital world, businesses and organizations are increasingly relying on efficient and engaging web presences to attract and retain customers. Landing pages play a crucial role in this process by providing a first impression and driving user engagement. Creating informative and appealing descriptions for landing pages is essential for web developers and designers to effectively convey their ideas to clients or teammates.

The "Automated Landing Page Description Generator" project aims to simplify this process by utilizing natural language processing techniques and state-of-the-art AI models. The project consists of a collection of Python classes that work together to generate comprehensive and human-readable descriptions for landing pages based on metadata. These descriptions include various components, such as text content, navigation bars, images, buttons, input fields, and iframes, along with their positions and colors.

By leveraging the capabilities of the GPT-3.5-turbo or GPT-4 model from OpenAI, the project is able to generate summaries of the landing pages that mimic a conversation between a user and an assistant. This interactive approach provides an intuitive way to understand the structure and content of the landing page, making it easier for software consulting firms or development teams to build the desired web presence.

With the Automated Landing Page Description Generator, web developers and designers can streamline their workflow, ensuring that they can effectively communicate their visions for landing pages and facilitate a seamless development process.


Project description
Convert website metadata into text prompts to serve as input for a generative AI model. The metadata includes information like descriptions and relative locations of images, videos, and text inputs on a website. The text prompts are to be verbal descriptions of the website.

Requirements

Python 3.7+

NLTK

OpenAI API (GPT-3.5-turbo)

### Demo
`Given metadata of following website:`<br>
![image](https://user-images.githubusercontent.com/71195307/233437412-fd8343a7-ee77-4034-88fe-f451f48cbccf.png)<br>
`Here is mockup output:`<br>
Sure, based on the information you provided, I would like the software consulting firm to build a landing page for a global pharmaceutical company that is research and development-driven and committed to improving the health of patients. The landing page should have an image of "HOME" in the top left corner, an input field for search in the right edge and above center, and an input field for entering email address in the bottom right corner. The landing page should also have four buttons: a white "SUBSCRIBE" button in the bottom and right center, a black "Cookies Settings" button below and to the right of center, a red "Reject All" button in the bottom and right center, and a red "Accept All Cookies" button in the bottom and right center. The landing page should not have any iframes.

`SpaceX:`<br>
![screenshot](https://user-images.githubusercontent.com/97744180/233453411-e03ba78d-54e1-4a15-9db7-c55ba417f0f9.png)
`Output:`<br>
Sure, here's a generic description of the landing page:
The landing page should have a navigation bar with the following options: Home, About Us, Products, Services, and Contact Us. The landing page should have an image of a rocket or spacecraft in the background to convey the company's focus on space technology. There should be a button to learn more about the company's mission and a button to view their products. The landing page should also have an input field for users to sign up for the company's newsletter. Finally, there should be an iframe displaying a video of a rocket launch or a spacecraft in orbit to further emphasize the company's expertise in space technology.

`Website for a theatre:`<br>
![screenshot](https://user-images.githubusercontent.com/97744180/233455058-956d8fea-ccd5-4115-869b-164d4547ca01.png)
`Output:`<br>
Sure, here's a generic description of the landing page:
The landing page should have a navigation bar with the following options: Home, About Us, Services, Portfolio, and Contact Us. The landing page should also have an input field of SEARCH in the top right corner.
The landing page should have an image section that showcases the performing arts. The images should be placed in a visually appealing manner and should be clickable.
The landing page should have an iframe section that displays the upcoming events. The iframes should be placed in a way that is easy to navigate and should be clickable
The landing page should have an input field that allows users to subscribe to the newsletter. The input field should be placed in a prominent location and should be easy to use. Overall, the landing page should be designed in a way that is visually appealing and easy to navigate. The content should be organized in a way that is easy to understand and should be optimized for search engines.

Contributing

We welcome contributions to improve and expand the functionality of the Automated Landing Page Description Generator. To contribute, please follow these steps:
Fork the repository.
Create a branch with a descriptive name, e.g., feature/new-generator-class.
Make your changes and commit them with informative commit messages.
Push your changes to the branch.
Create a pull request to the main repository.

