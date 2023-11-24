<template>
  <div class="file-upload">
    <div
      class="drop-area"
      :class="{ 'drag-over': isDragging }"
      id="box"
      @dragover.prevent="handleDragOver"
      @dragleave.prevent="handleDragLeave"
      @drop.prevent="handleDrop"
      @click.prevent="openFilePicker"
    >
      <p>Перетащите файлы сюда или кликните для выбора файла</p>
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
          <div class="buttons">
            <button
              @click="removeFile(index)"
              class="btn-rem btn-hover color-1"
            >
              Удалить
            </button>
          </div>
        </li>
      </ul>
    </div>
    <div class="buttons">
      <button
        @click="submitFiles"
        :disabled="filesToUpload.length === 0"
        class="btn-up btn-hover color-2"
      >
        Отправить
      </button>
    </div>
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
      isDragging: false,
    };
  },
  methods: {
    handleDragOver(event) {
      event.preventDefault();
      this.isDragging = true;
    },
    handleDrop(event) {
      event.preventDefault();
      this.isDragging = false;

      const files = event.dataTransfer.files;
      this.filesToUpload = this.filesToUpload.concat(Array.from(files));
    },
    handleDragLeave() {
      this.isDragging = false;
    },
    openFilePicker() {
      const inputElement = document.createElement("input");
      inputElement.type = "file";
      inputElement.multiple = true;
      inputElement.click();

      inputElement.addEventListener("change", (event) => {
        const files = event.target.files;
        this.filesToUpload = this.filesToUpload.concat(Array.from(files));
      });
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
          }
        );

        console.log("Upload successful", response.data);

        this.uploadedFiles = this.uploadedFiles.concat(this.filesToUpload);
        this.filesToUpload = [];
      } catch (error) {
        console.error("Error uploading files", error);
        console.log("Response data:", error.response.data);
      }
    },
  },
};
</script>

<style scoped>
p {
  font-size: 30px;
  font-weight: 600;
}

.file-upload {
  max-width: 800px;
  margin: auto;
  margin-top: 10vh;
}

#box {
  height: 35vh;
  margin-bottom: 80px;
  color: #000000;
  font-family: "Raleway";
  font-size: 1rem;
}

.drop-area {
  border: 2px dashed #ccc;
  padding: 20px;
  height: 50vh;
  text-align: center;
  cursor: pointer;
  border-radius: calc(2 * var(--borderWidth));
  margin-bottom: 20px;
  --borderWidth: 10px;
  position: relative;
  box-shadow: 0px 40px 100px 60px rgba(245, 51, 51, 0.25);
}

.drop-area.drag-over {
  background-color: #f1f1f1; /* Белый фон при перетаскивании */
  box-shadow: 0px 40px 100px 60px rgba(74, 245, 51, 0.25);
}

.drop-area:after {
  content: "";
  position: absolute;
  top: calc(-1 * var(--borderWidth));
  left: calc(-1 * var(--borderWidth));
  height: calc(100% + var(--borderWidth) * 2);
  width: calc(100% + var(--borderWidth) * 2);
  background: linear-gradient(
    60deg,
    #f79533,
    #f37055,
    #ef4e7b,
    #a166ab,
    #5073b8,
    #1098ad,
    #07b39b,
    #6fba82
  );
  border-radius: calc(2 * var(--borderWidth));
  z-index: -1;
  animation: animatedgradient 3s ease alternate infinite;
  background-size: 300% 300%;
}

@keyframes animatedgradient {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
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
  justify-content: center;
  align-items: center;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 5px;
}

.file-name {
  margin-right: 10px;
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

.buttons {
  text-align: center;
}

.btn-rem {
  width: 150px;
  height: 40px;
  font-size: 16px;
  color: #000;
}

.btn-up {
  width: 200px;
  height: 55px;
  font-weight: 600;

  font-size: 20px;
  color: #000;
}

.btn-hover {
  cursor: pointer;
  margin: 20px;
  text-align: center;
  border: none;
  background-size: 300% 100%;
  border-radius: 50px;
  -o-transition: all 0.4s ease-in-out;
  -webkit-transition: all 0.4s ease-in-out;
  transition: all 0.4s ease-in-out;
}

.btn-hover:hover {
  background-position: 100% 0;
  -o-transition: all 0.4s ease-in-out;
  -webkit-transition: all 0.4s ease-in-out;
  transition: all 0.4s ease-in-out;
}

.btn-hover:focus {
  outline: none;
}

.btn-hover.color-2 {
  background-image: linear-gradient(
    to right,
    #0ba360,
    #3cba92,
    #30dd8a,
    #2bb673
  );
  box-shadow: 0 40px 100px 0 rgba(23, 168, 108, 0.75);
}

.btn-hover.color-1 {
  background-image: linear-gradient(
    to right,
    #ed6ea0,
    #ec8c69,
    #f7186a,
    #fbb03b
  );
  box-shadow: 0 4px 15px 0 rgba(236, 116, 149, 0.75);
}
</style>
