#include "filesetter.h"

#include <QFileDialog>
#include <QGridLayout>
#include <iostream>

using namespace std;

FileSetter::FileSetter(QMainWindow *parent, const QString& labelText, const QString& filter)
{
    mainWindow = parent;
    this->filter = filter;
    label = new QLabel(labelText);
    text = new QLineEdit();
    button = new QPushButton("...");
    button->setMaximumWidth(60);
    connect(button, SIGNAL(clicked()), this, SLOT(selectFile()));

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(label,  0, 0, 1, 1);
    layout->addWidget(text,   0, 1, 1, 1);
    layout->addWidget(button, 0, 2, 1, 1);

    setLayout(layout);
}

void FileSetter::setPath(const QString& path)
{
    filename = path;
    text->setText(path);
}

void FileSetter::selectFile()
{
    QString default_path = text->text();
    if (default_path.length() == 0)
        default_path = QDir::homePath();

    QString path = QFileDialog::getOpenFileName(mainWindow, label->text(), default_path, filter);
    if (path.length() > 0) {
        filename = path;
        text->setText(filename);
        emit fileSet(filename);
    }
}

void FileSetter::trimHeight()
{
    setMaximumHeight(label->fontMetrics().boundingRect("Xy").height() * 4);
}
