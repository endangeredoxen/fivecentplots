FROM ubuntu:20.04

RUN apt update
RUN apt-get install -y python3.8
RUN apt install -y git
RUN git clone https://github.com/endangeredoxen/fivecentplots.git && cd fivecentplots && git checkout v0.5.0-beta
RUN apt-get install -y pip
RUN cd fivecentplots && pip3 install .[test]
