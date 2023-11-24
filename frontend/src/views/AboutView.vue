<template>
  <div class="about">
    <Textimg :videoURL="videoURL[0]" :title="title[0]" shadowType="light" />
    <Textimg :videoURL="videoURL[1]" :title="title[1]" shadowType="dark" />
  </div>
</template>

<script>
import axios from "axios";
import Textimg from "@/components/Text-img.vue";

export default {
  name: "About",
  data() {
    return {
      videoURL: [],
      title: ["До корректировки", "После корректировки"],
      error: null,
    };
  },

  components: {
    Textimg,
  },

  mounted() {
    axios
      .get("http://26.234.143.237:8000/video")
      .then((response) => {
        if (response.status === 200) {
          this.videoURL = [response.data.VideoURL1, response.data.VideoURL2];
        } else {
          this.error = "Ошибка при получении данных: неверный статус ответа";
        }
      })
      .catch((error) => {
        console.error("Ошибка при получении данных:", error);
        this.error = "Ошибка при получении данных: " + error.message;
      });
  },
};
</script>

<style>
.about {
  display: flex;
  justify-content: space-around;
}

.buttons {
  text-align: center;
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
</style>
