from setuptools import find_packages, setup 
from typing import List
#this will find all the packages in the current directory/that are used in our ML project

hyphen_e_dot = '-e .'
def get_requirements(file_path)->List[str]:
    '''
    This function will read the requirements.txt file and return a list of packages
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements = [req.replace('\n',' ') for req in requirements]

    if hyphen_e_dot in requirements:
        requirements.remove(hyphen_e_dot)
    
    return requirements



setup(
    name = 'ml_e2e_project',
    version='0.0.1',
    author='Siddharth',
    author_email='sarkarsiddhartha758@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),

) #this will create a setup.py file