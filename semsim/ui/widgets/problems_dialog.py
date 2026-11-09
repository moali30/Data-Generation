"""A dialog listing template problems, ready to copy back to the assistant."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from semsim.ui.i18n import translator


class ProblemsDialog(QDialog):
    """Show a list of problems with a button that copies a fix-it request."""

    def __init__(self, title: str, intro: str, body: str, copy_text: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 420)
        self.copy_text = copy_text
        layout = QVBoxLayout(self)
        message = QLabel(intro)
        message.setWordWrap(True)
        layout.addWidget(message)
        self.list = QPlainTextEdit(body)
        self.list.setReadOnly(True)
        layout.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        self.copy_button = QPushButton(translator.t("ai.copy_problems"))
        self.copy_button.clicked.connect(self._copy)
        buttons.addWidget(self.copy_button)
        buttons.addStretch(1)
        close = QPushButton(translator.t("ai.close"))
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _copy(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.copy_text)
        self.copy_button.setText(translator.t("ai.copied"))


__all__ = ["ProblemsDialog"]
