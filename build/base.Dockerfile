# start by pulling the python image
FROM python:3.12

# install the dependencies and packages in the requirements file
RUN pip install -r ./requirements.txt