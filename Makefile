PANDOC_FLAGS = --pdf-engine=typst -V margin-x=2cm -V margin-y=2.5cm
THESIS_SOURCE = thesis/THESIS.md
THESIS_PDF = thesis/THESIS.pdf
THESIS_HTML = thesis/THESIS.html

.PHONY: all build clean

all: build

build: $(THESIS_PDF) $(THESIS_HTML)

$(THESIS_PDF): $(THESIS_SOURCE)
	pandoc $< -o $@ $(PANDOC_FLAGS)

$(THESIS_HTML): $(THESIS_SOURCE)
	pandoc $< -o $@

clean:
	rm -f $(THESIS_PDF) $(THESIS_HTML)
