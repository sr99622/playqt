/*******************************************************************************
* messagebox.cpp
*
* Copyright (c) 2020 Stephen Rhodes
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*******************************************************************************/

#include "messagedialog.h"

#include <QPushButton>
#include <QDialogButtonBox>
#include <QVBoxLayout>
#include <QCloseEvent>
#include <QClipboard>
#include <QApplication>

MessageDialog::MessageDialog(QMainWindow *parent) : PanelDialog(parent)
{
    setWindowTitle(tr("Messages"));
    message = new QTextEdit();
    QPushButton *clearButton = new QPushButton(tr("Clear"), this);
    connect(clearButton, SIGNAL(clicked()), this, SLOT(clear()));
    QPushButton *closeButton = new QPushButton(tr("Close"), this);
    connect(closeButton, SIGNAL(clicked()), this, SLOT(close()));
    QPushButton *copyButton = new QPushButton(tr("Copy"), this);
    connect(copyButton, SIGNAL(clicked()), this, SLOT(copy()));
    QDialogButtonBox *buttonBox = new QDialogButtonBox(Qt::Horizontal, this);
    buttonBox->addButton(copyButton, QDialogButtonBox::ActionRole);
    buttonBox->addButton(clearButton, QDialogButtonBox::ActionRole);
    buttonBox->addButton(closeButton, QDialogButtonBox::RejectRole);
    QVBoxLayout *dlgLayout = new QVBoxLayout();
    dlgLayout->addWidget(message);
    dlgLayout->addWidget(buttonBox);
    setLayout(dlgLayout);
    message->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);

    defaultWidth = 400;
    defaultHeight = 400;
    settingsKey = "MessageBox/size";

}

void MessageDialog::clear()
{
    message->setText("");
}

void MessageDialog::copy()
{
    fprintf(stderr, "this\n");
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(message->toPlainText());
}

