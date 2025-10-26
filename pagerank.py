import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    N = len(corpus)
    pages = corpus[page]
    distribution = dict()

    # If a page has no links in it, each page in the corpus is equally likely.
    if len(pages) == 0:
        return {p: 1 / N for p in corpus}

    # Initially dstribute 1 - D equally to all
    for corp in corpus:
        distribution[corp] = (1 - damping_factor) / N

    # Distribute the rest of the probability
    for page in pages:
        distribution[page] = distribution.get(page) + damping_factor / len(pages)
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    samples = []

    # Make a new sample depending upon the page (Previous sample)
    def sample(page):
        distribution = transition_model(corpus, page, damping_factor)
        return random.choices(
            list(distribution.keys()), weights=list(distribution.values())
        )[0]

    # Choosing a sample based on the previous sample
    for i in range(n):
        if samples == []:
            samples.append(sample(random.choice(list(corpus.keys()))))
        else:
            samples.append(sample(samples[i - 1]))

    # Count the samples using a dictonary
    pagerank = dict()
    for s in samples:
        if pagerank.get(s, None) is None:
            pagerank[s] = 1
        else:
            pagerank[s] += 1

    # Normalize the count and return the distribution
    return {p: w / n for p, w in pagerank.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initial distribution of ranks of pages
    pages = list(corpus.keys())
    n = len(pages)
    distribution = dict()
    for page in pages:
        distribution[page] = 1 / n
    copy_distribution = distribution.copy()
    while True:
        for p in pages:
            contribs = []
            for u, links in corpus.items():
                if p in links:
                    if len(links) > 0:
                        contribs.append(distribution[u] / len(links))
                if len(links) == 0:
                    contribs.append(distribution[u] / n)
            val = (1 - damping_factor) / n + damping_factor * sum(contribs)
            distribution[p] = val
        total = sum(distribution.values())
        for p in distribution:
            distribution[p] /= total
        diffs = [
            abs(distribution[p] - copy_distribution[p])
            for p in list(distribution.keys())
        ]
        if max(diffs) < 0.001:
            break

        copy_distribution = distribution.copy()
    return distribution


if __name__ == "__main__":
    main()
