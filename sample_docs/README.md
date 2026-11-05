Put course PDFs here that should be built into the hosted app for every visitor.

After adding or changing PDFs, rebuild the index and push:

    pip install -r requirements-dev.txt
    python -m scripts.build_sample_index
    git add sample_docs public/static/sample_index.json && git commit -m "Update built-in documents" && git push

Anything in this folder becomes public (GitHub and the website), so only add
material you are allowed to share.
