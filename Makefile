PANDOC_FLAGS = --pdf-engine=typst -V margin-x=2cm -V margin-y=2.5cm

.PHONY: all build clean

all: build

build: ARXIV-PAPER.pdf PHD-THESIS.pdf

%.pdf: %.md
	pandoc $< -o $@ $(PANDOC_FLAGS)

clean:
	rm -f ARXIV-PAPER.pdf PHD-THESIS.pdf
