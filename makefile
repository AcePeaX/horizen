compile:
	( \
       source .venv/bin/activate; \
       python3 scripts/lib/utils/compile.py; \
    )
sa-train:
	( \
       source .venv/bin/activate; \
       python3 scripts/bin/sa-model.py; \
    )
test:
	( \
       source .venv/bin/activate; \
       python3 scripts/bin/test-model.py; \
    )
cook-fineweb:
	( \
       source .venv/bin/activate; \
       python3 scripts/bin/cooking/fineweb-edu-score-2.py; \
    )