FROM pytorch/pytorch

RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev

WORKDIR /home/chekalina/
COPY requirements.txt /home/chekalina/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install spacy
RUN python3.6 -m spacy download en_core_web_sm
RUN python3.6 -m spacy download en_core_web_md
RUN pip3 install -r requirements.txt
CMD /home/chekalina/cqas/start.sh
