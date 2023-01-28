FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive

# Install python 3.8
RUN apt update
RUN apt-get install -y python3.8
RUN apt-get install -y pip

# Install git
RUN apt install -y git

# Clone fcp
RUN git clone https://github.com/endangeredoxen/fivecentplots.git && cd fivecentplots && git checkout v0.5.3

# Install fcp
RUN cd fivecentplots && pip3 install .[test]

# Clean up fonts
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt install -y msttcorefonts -qq
RUN rm ~/.cache/matplotlib -rf
