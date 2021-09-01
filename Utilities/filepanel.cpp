#include "filepanel.h"
#include "mainwindow.h"

FilePanel::FilePanel(QMainWindow *parent, const QString& name, const QString& defaultPath) : QWidget(parent)
{
    mainWindow = parent;
    this->name = name;
    this->defaultPath = defaultPath;

    directorySetter = new DirectorySetter(mainWindow, "");
    model = new QFileSystemModel();
    model->setReadOnly(false);
    tree = new TreeView(this);
    tree->setModel(model);

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(directorySetter,      0, 0, 1, 1);
    layout->addWidget(tree,                 1, 0, 1, 1);
    setLayout(layout);

    QString path = MW->settings->contains(getDirKey()) ? MW->settings->value(getDirKey()).toString() : defaultPath;
    directorySetter->setPath(path);
    model->setRootPath(path);
    tree->setRootIndex(model->index(path));
    connect(directorySetter, SIGNAL(directorySet(const QString&)), this, SLOT(setDirectory(const QString&)));

    if (MW->settings->contains(getHeaderKey()))
        tree->header()->restoreState(MW->settings->value(getHeaderKey()).toByteArray());
    tree->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(tree, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
    connect(tree, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(doubleClicked(const QModelIndex&)));
    connect(tree->header(), SIGNAL(sectionResized(int, int, int)), this, SLOT(headerChanged(int, int, int)));
    connect(tree->header(), SIGNAL(sectionMoved(int, int, int)), this, SLOT(headerChanged(int, int, int)));

    menu = new QMenu("Context Menu", this);
    QAction *remove = new QAction("Delete", this);
    QAction *rename = new QAction("Rename", this);
    QAction *info = new QAction("Info", this);
    QAction *play = new QAction("Play", this);
    connect(remove, SIGNAL(triggered()), this, SLOT(remove()));
    connect(rename, SIGNAL(triggered()), this, SLOT(rename()));
    connect(info, SIGNAL(triggered()), this, SLOT(info()));
    connect(play, SIGNAL(triggered()), this, SLOT(play()));
    menu->addAction(remove);
    menu->addAction(rename);
    menu->addAction(info);
    menu->addAction(play);

    connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));
}

QString FilePanel::getDirKey() const
{
    return name + "/dir";
}

QString FilePanel::getHeaderKey() const
{
    return name + "/header";
}

void FilePanel::setDirectory(const QString& path)
{
    directorySetter->setPath(path);
    model->setRootPath(path);
    tree->setRootIndex(model->index(path));
    MW->settings->setValue(getDirKey(), path);
}

void FilePanel::autoSave()
{
    if (changed) {
        MW->settings->setValue(getHeaderKey(), tree->header()->saveState());
        changed = false;
    }
}

void FilePanel::doubleClicked(const QModelIndex& index)
{
    if (index.isValid()) {
        QFileInfo info = model->fileInfo(index);
        if (info.isDir()) {
            bool expanded = tree->isExpanded(index);
            tree->setExpanded(index, !expanded);
        }
        else {
            MW->control()->play();
        }
    }
}

void FilePanel::play()
{
    doubleClicked(tree->currentIndex());
}

void FilePanel::headerChanged(int arg1, int arg2, int arg3)
{
    if (isVisible())
        changed = true;
}

void FilePanel::showContextMenu(const QPoint &pos)
{
    QModelIndex index = tree->indexAt(pos);
    if (index.isValid()) {
        menu->exec(mapToGlobal(pos));
    }
}

void FilePanel::remove()
{
    QModelIndex index = tree->currentIndex();
    if (!index.isValid())
        return;

    int ret = QMessageBox::warning(this, "playqt",
                                   "You are about to delete this file.\n"
                                   "Are you sure you want to continue?",
                                   QMessageBox::Ok | QMessageBox::Cancel);

    if (ret == QMessageBox::Ok)
        QFile::remove(model->filePath(tree->currentIndex()).toLatin1().data());
}

void FilePanel::rename()
{
    QModelIndex index = tree->currentIndex();
    if (index.isValid())
        tree->edit(index);
}
void FilePanel::info()
{
    QModelIndex index = tree->currentIndex();
    if (!index.isValid())
        return;

    QString filename = model->filePath(tree->currentIndex());
    AVFormatContext *fmt_ctx = nullptr;
    AVStream *video;
    AVStream *audio;
    int video_stream;
    int audio_stream;

    try {
        av.ck(avformat_open_input(&fmt_ctx, filename.toLatin1().data(), NULL, NULL), AOI);
        av.ck(avformat_find_stream_info(fmt_ctx, NULL), AFSI);
    }
    catch (AVException *e) {
        emit msg(QString("Unable to open format context %1: %2\n").arg(av.tag(e->cmd_tag), e->error_text));
        return;
    }

    try {
        av.ck(audio_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0), AFBS);
        audio = fmt_ctx->streams[audio_stream];
        QString str = "File audio parameters\n";

        char buf[16];

        QString codec_str;
        const AVCodecDescriptor *cd = avcodec_descriptor_get(audio->codecpar->codec_id);
        if (cd) {
            QTextStream(&codec_str) << "codec_name: " << cd->name << "\n"
                                    << "codec_long_name: " << cd->long_name << "\n";
        }
        else {
            QTextStream(&codec_str) << "Uknown codec" << "\n";
        }

        if (fmt_ctx->metadata == NULL) {
            str.append("\nmetadata is NULL\n");
        }
        else {
            QTextStream(&str) << "\n";
            AVDictionaryEntry *t = NULL;
            while (t = av_dict_get(fmt_ctx->metadata, "", t, AV_DICT_IGNORE_SUFFIX)) {
                QTextStream(&codec_str) << t->key << " : " << t->value << "\n";
            }
        }

        QTextStream(&str)
            << "filename: " << filename << "\n"
            << codec_str

            << "format: " << fmt_ctx->iformat->long_name << " (" << fmt_ctx->iformat->name << ")\n"
            << "flags: " << fmt_ctx->iformat->flags << "\n"
            << "extradata_size: " << audio->codecpar->extradata_size << "\n"
            << "codec time_base:  " << audio->codec->time_base.num << " / " << audio->codec->time_base.den << "\n"
            << "audio time_base: " << audio->time_base.num << " / " << audio->time_base.den << "\n"
            << "codec framerate: " << audio->codec->framerate.num << " / " << audio->codec->framerate.den << "\n"
            << "ticks_per_frame: " << audio->codec->ticks_per_frame << "\n"
            << "bit_rate: " << fmt_ctx->bit_rate << "\n"
            << "codec framerate: " << av_q2d(audio->codec->framerate) << "\n"
            << "start_time: " << fmt_ctx->start_time * av_q2d(av_get_time_base_q()) << "\n"
            << "duration: " << fmt_ctx->duration * av_q2d(av_get_time_base_q()) << "\n";

        emit msg(str);
        MW->messageDialog->show();
    }
    catch (AVException *e) {
        emit msg(QString("Unable to process audio stream %1: %2\n").arg(av.tag(e->cmd_tag), e->error_text));
    }

    try {
        av.ck(video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0), AFBS);
        video = fmt_ctx->streams[video_stream];

        QString str = "File video parameters\n";

        QString codec_str;
        const AVCodecDescriptor *cd = avcodec_descriptor_get(video->codecpar->codec_id);
        if (cd) {
            QTextStream(&codec_str) << "codec_name: " << cd->name << "\n"
                                    << "codec_long_name: " << cd->long_name << "\n";
        }
        else {
            QTextStream(&codec_str) << "Uknown codec" << "\n";
        }

        if (fmt_ctx->metadata == NULL) {
            str.append("\nmetadata is NULL\n");
        }
        else {
            QTextStream(&str) << "\n";
            AVDictionaryEntry *t = NULL;
            while (t = av_dict_get(fmt_ctx->metadata, "", t, AV_DICT_IGNORE_SUFFIX)) {
                QTextStream(&codec_str) << t->key << " : " << t->value << "\n";
            }
        }

        QTextStream(&str)
            << "filename: " << filename << "\n"
            << "pixel format: " << av_get_pix_fmt_name((AVPixelFormat)video->codecpar->format) << "\n"
            << codec_str
            << "format: " << fmt_ctx->iformat->long_name << " (" << fmt_ctx->iformat->name << ")\n"
            << "flags: " << fmt_ctx->iformat->flags << "\n"
            << "extradata_size: " << video->codecpar->extradata_size << "\n"
            << "codec time_base:  " << video->codec->time_base.num << " / " << video->codec->time_base.den << "\n"
            << "video time_base: " << video->time_base.num << " / " << video->time_base.den << "\n"
            << "codec framerate: " << video->codec->framerate.num << " / " << video->codec->framerate.den << "\n"
            << "gop_size: " << video->codec->gop_size << "\n"
            << "ticks_per_frame: " << video->codec->ticks_per_frame << "\n"
            << "bit_rate: " << fmt_ctx->bit_rate << "\n"
            << "codec framerate: " << av_q2d(video->codec->framerate) << "\n"
            << "start_time: " << fmt_ctx->start_time * av_q2d(av_get_time_base_q()) << "\n"
            << "duration: " << fmt_ctx->duration * av_q2d(av_get_time_base_q()) << "\n"
            << "width: " << video->codecpar->width << "\n"
            << "height: " << video->codecpar->height << "\n";

        emit msg(str);
        MW->messageDialog->show();
    }
    catch (AVException *e) {
        emit msg(QString("Unable to process video stream %1: %2\n").arg(av.tag(e->cmd_tag), e->error_text));
    }

    if (fmt_ctx != nullptr)
        avformat_close_input(&fmt_ctx);
}

TreeView::TreeView(QWidget *parent) : QTreeView(parent)
{

}


void TreeView::mouseDoubleClickEvent(QMouseEvent *event)
{
    emit doubleClicked(indexAt(event->pos()));
}
