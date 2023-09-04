from setuptools import find_packages, setup
from typing import List

Hypen_dot_e="-e ."

def get_requrements(filepath:str)-> List[str]:
    requirements=[]

    with open(filepath) as file_obj:
        requirements=file_obj.readlines()
        requirements=[ i.replace("\n","") for i  in requirements]

        if Hypen_dot_e in requirements:
            requirements.remove(Hypen_dot_e)
             

setup(name='Loan Aproval Prediction',
      version='0.0.1',
      description='Loan Approval',
      author='Nikhil Paulzagde',
      author_email='nickpaulzagde@gmail.com',
      url='https://github.com/NikhilPaulzagde/Data-Science-End-to-End-Project/tree/main/Loan%20Approval%20Predictor',
      packages=find_packages(),
      install_requires=get_requrements("requirements.txt")
    )