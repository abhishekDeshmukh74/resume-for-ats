.PHONY: dev backend frontend install

dev:
	@echo "Starting backend and frontend..."
	@start "backend" cmd /k "cd backend && uvicorn backend.main:app --reload"
	@cd frontend && npm run dev

backend:
	uvicorn backend.main:app --reload

frontend:
	cd frontend && npm run dev

install:
	pip install -r backend/requirements.txt
	cd frontend && npm install
