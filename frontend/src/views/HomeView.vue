<template>
  <div class="home">
    <Drop />
  </div>
  <div class="about">
    <Textimg :videoURL="videoURL[0]" :title="title[0]" shadowType="red" />
    <Textimg :videoURL="videoURL[1]" :title="title[1]" shadowType="green" />
  </div>
</template>

<script>
import axios from "axios";
import Drop from "@/components/Drop.vue";
import Textimg from "@/components/Text-img.vue";

export default {
  data() {
    return {
      videoURL: [],
      title: ["До корректировки", "После корректировки"],
      error: null,
    };
  },

  components: {
    Drop,
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

  margin-top: 5vh;
}

</style>
