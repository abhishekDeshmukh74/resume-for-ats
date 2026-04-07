.PHONY: dev backend frontend install

dev:
	@echo "Starting backend and frontend..."
	@source .venv/bin/activate && uvicorn backend.main:app --reload --port 8000 &
	@cd frontend && npm run dev

backend:
	source .venv/bin/activate && uvicorn backend.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

install:
	pip install -r backend/requirements.txt
	cd frontend && npm install
