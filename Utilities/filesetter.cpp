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
    button->setMaximumWidth(30);
    connect(button, SIGNAL(clicked()), this, SLOT(selectFile()));

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(label,  0, 0, 1, 1);
    layout->addWidget(text,   0, 1, 1, 1);
    layout->addWidget(button, 0, 2, 1, 1);

    layout->setContentsMargins(0, 0, 0, 0);

    setLayout(layout);

    setContentsMargins(0, 0, 0, 0);
}

void FileSetter::setPath(const QString& path)
{
    filename = path;
    text->setText(path);
}

void FileSetter::setPath()
{
    text->setText(filename);
}

void FileSetter::selectFile()
{
    QString default_path = text->text();
    if (default_path.length() == 0)
        default_path = QDir::homePath();

    QString path = QFileDialog::getOpenFileName(mainWindow, label->text(), default_path, filter);
    cout << "test 1" << endl;
    if (path.length() > 0) {
        filename = path;
        text->setText(filename);
        emit fileSet(filename);
    }
}

