# Senior Project 2024 Group 1

## Abstract
Create a Machine Learning (ML) Model that will take active data from a host network and give everything a risk level, that will intern give a rating of possible increased activity of bad actors. The system will be taught by previous network data and MITRE framework terminology. The risk levels will be based on a risk assessment of the information gathered and its impact on the host network. It will be displayed as a “Health Level”, where the higher the percentage, the better the network is protected. IE: A Max score is a network in which bad actors are unable to get in, or exfil data.

## Python Dependencies

- [ ] tkinter
- [ ] pandas
- [ ] matplotlib.plot
- [ ] sklearn (formerly scikit-learn)
- [ ] joblib
- [ ] graphviz

## Installation

- Main function located within .\senior-project-2024-group-1\python\main.py
- Simply run using python ie `python3 .\senior-project-2024-group-1\python\main.py`
- A command prompt shell should open alongside the tkinter main window which houses the interface

## Usage
- Run Model Training - begins the ML training process. Asks for data as input with which to train and asks the User whether the model should be saved as a .joblib file.
- Select Pretrained Model - brings up a file selection window to take a .joblib file. 
- Debug Checkbox - runs the program in debug mode
- Predict - takes in unlabeled data and produces labels in the order of top to bottom of the input file. Must select pretrained model first.
- Clear output text - clears the textbox and removes any chosen pretrained model.

## Support
Cry

## Roadmap
humungous things in the works

## Authors and acknowledgment

Project by:
- Daniel Hassler
- Samuel Hibbard
- Ben Hixon
- Chuck Kudzmas
- Ryan McShane
- Alek Wasserman

Thanks to Professor Cordano

## Project status
Ongoing