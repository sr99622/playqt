#include "filepanel.h"
#include "mainwindow.h"

FilePanel::FilePanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    directorySetter = new DirectorySetter(mainWindow, "");
    directorySetter->trimHeight();
    //directorySetter->setPath(QStandardPaths::writableLocation(QStandardPaths::MoviesLocation));
    model = new QFileSystemModel();
    //model->setRootPath(QStandardPaths::writableLocation(QStandardPaths::MoviesLocation));
    tree = new QTreeView(mainWindow);
    tree->setModel(model);
    //tree->setRootIndex(model->index(QStandardPaths::writableLocation(QStandardPaths::MoviesLocation)));

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(directorySetter,      0, 0, 1, 1);
    layout->addWidget(tree,                 1, 0, 1, 1);
    setLayout(layout);

    connect(directorySetter, SIGNAL(directorySet(const QString&)), this, SLOT(setDirectory(const QString&)));
    connect(tree, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(doubleClicked(const QModelIndex&)));
}

void FilePanel::setDirectory(const QString& path)
{
    //cout << "VideoPanel::setDirectory " << path.toStdString() << endl;
    directorySetter->setPath(path);
    model->setRootPath(path);
    tree->setRootIndex(model->index(path));
}

void FilePanel::doubleClicked(const QModelIndex& index)
{
    cout << "VideoPanel::doubleClicked" << endl;
    if (index.isValid()) {
        QString str = model->filePath(index);
        cout << str.toStdString() << endl;
        //MW->filename = str;
        MW->co->input_filename = av_strdup(str.toLatin1().data());
        MW->runLoop();
    }
}
