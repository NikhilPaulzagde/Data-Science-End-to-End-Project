from setuptools import setup, find_packages
from typing import List

Hypen_e="-e ."

def get_requirements()-> List[str]:
    with open("requirements.txt") as file:
        requirement_list=file.readlines()
        requirement_list=[ file.replace("\n","") for file in requirement_list]

        if Hypen_e in requirement_list:
            requirement_list.remove(Hypen_e)

        return requirement_list





setup(name='Delivery Time Prediction',
      version='0.0.1',
      description='Machine Learning Project',
      author='Nikhil Paulzagde',
      author_email='nickpaulzagde@gmail.com',
      url='https://bit.ly/2U6KLvV',
      packages=find_packages(),
      install_requires= get_requirements()

     )


