<template>
  <div class="file-upload">
    <div
      class="drop-area"
      @dragover.prevent="handleDragOver"
      @drop.prevent="handleDrop"
    >
      Перетащите файлы сюда
    </div>
    <div v-if="filesToUpload.length > 0" class="preview-container">
      <h3>Предварительный просмотр:</h3>
      <ul class="file-list">
        <li
          v-for="(file, index) in filesToUpload"
          :key="index"
          class="file-item"
        >
          <img src="@/assets/image-icon.svg" width="30" />
          <span class="file-name">{{ file.name }}</span>
          <button @click="removeFile(index)" class="remove-button">
            Удалить
          </button>
        </li>
      </ul>
    </div>
    <button
      @click="submitFiles"
      :disabled="filesToUpload.length === 0"
      class="upload-button"
    >
      Отправить
    </button>
    <ul v-if="uploadedFiles.length" class="uploaded-list">
      <li v-for="file in uploadedFiles" :key="file.name" class="uploaded-item">
        {{ file.name }}
      </li>
    </ul>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      uploadedFiles: [],
      filesToUpload: [],
    };
  },
  methods: {
    handleDragOver(event) {
      event.preventDefault();
    },
    handleDrop(event) {
      event.preventDefault();

      const files = event.dataTransfer.files;
      this.filesToUpload = this.filesToUpload.concat(Array.from(files));
    },
    removeFile(index) {
      this.filesToUpload.splice(index, 1);
    },
    async submitFiles() {
      if (this.filesToUpload.length === 0) {
        alert("Выберите файлы для отправки.");
        return;
      }

      console.log("Files to upload:", this.filesToUpload);

      const formData = new FormData();

      for (let i = 0; i < this.filesToUpload.length; i++) {
        const file = this.filesToUpload[i];
        formData.append("file", file);
      }

      try {
        const response = await axios.post(
          "http://26.234.143.237:8000/archive_extract",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          },
        );

        console.log("Upload successful", response.data);

        this.uploadedFiles = this.uploadedFiles.concat(this.filesToUpload);
        this.filesToUpload = [];
      } catch (error) {
        console.error("Error uploading files", error);
        console.log('Response data:', error.response.data);
      }
    },
  },
};
</script>

<style scoped>
.file-upload {
  max-width: 800px;
  margin: auto;
}

.drop-area {
  border: 2px dashed #ccc;
  padding: 20px;
  height: 50vh;
  text-align: center;
  cursor: pointer;
  background-color: #f9f9f9;
  border-radius: 20px;
  margin-bottom: 20px;
}

.drop-area:hover {
  border: 2px dashed #000000;
}

.drop-area:active {
  border: 2px dashed #000000;
}

.preview-container {
  margin-bottom: 20px;
}

.file-list {
  list-style: none;
  padding: 0;
}

.file-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #fff;
  border: 1px solid #ddd;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 5px;
}

.file-name {
  flex-grow: 1;
  margin-right: 10px;
}

.remove-button {
  background-color: #e74c3c;
  color: #fff;
  border: none;
  padding: 5px 10px;
  cursor: pointer;
  border-radius: 3px;
}

.upload-button {
  background-color: #3498db;
  color: #fff;
  border: none;
  padding: 10px 20px;
  cursor: pointer;
  border-radius: 5px;
  font-size: 16px;
}

.uploaded-list {
  list-style: none;
  padding: 0;
}

.uploaded-item {
  background-color: #2ecc71;
  color: #fff;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 5px;
}
</style>
