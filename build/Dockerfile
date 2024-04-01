# start by pulling the python image
FROM thesmile-base:latest

# switch working directory
WORKDIR /app

# copy every content from the local file to the image
COPY . /app

# download models
RUN python build/download.py

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py" ]