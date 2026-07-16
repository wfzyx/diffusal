PANDOC_FLAGS = --pdf-engine=typst -V margin-x=2cm -V margin-y=2.5cm

.PHONY: all build clean

all: build

build: THESIS.pdf

THESIS.pdf: THESIS.md
	pandoc $< -o $@ $(PANDOC_FLAGS)

clean:
	rm -f THESIS.pdf
