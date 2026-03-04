SERVICE = pmt-analysis.service
SERVICE_FILE = $(SERVICE)
INSTALL_DIR = /etc/systemd/system

.PHONY: help install uninstall start stop restart status logs enable disable reload

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install    Copy service file and reload systemd"
	@echo "  uninstall  Stop, disable, and remove the service"
	@echo "  start      Start the service"
	@echo "  stop       Stop the service"
	@echo "  restart    Restart the service"
	@echo "  status     Show service status"
	@echo "  logs       Tail live logs (Ctrl+C to exit)"
	@echo "  enable     Enable service to start on boot"
	@echo "  disable    Disable service from starting on boot"
	@echo "  reload     Reload systemd daemon"

install:
	@echo "Installing $(SERVICE)..."
	sudo cp $(SERVICE_FILE) $(INSTALL_DIR)/$(SERVICE)
	sudo systemctl daemon-reload
	@echo "Done. Run 'make enable start' to activate."

uninstall:
	@echo "Uninstalling $(SERVICE)..."
	sudo systemctl stop $(SERVICE) 2>/dev/null || true
	sudo systemctl disable $(SERVICE) 2>/dev/null || true
	sudo rm -f $(INSTALL_DIR)/$(SERVICE)
	sudo systemctl daemon-reload
	@echo "Done."

start:
	chmod +x run_analysis.sh
	sudo systemctl start $(SERVICE)

stop:
	sudo systemctl stop $(SERVICE)

restart:
	sudo systemctl restart $(SERVICE)

status:
	systemctl status $(SERVICE)

logs:
	journalctl -u $(SERVICE) -f

enable:
	sudo systemctl enable $(SERVICE)

disable:
	sudo systemctl disable $(SERVICE)

reload:
	sudo systemctl daemon-reload