<template>
  <div class="about">
    <Textimg :imgURL="imgURL[0]" :title="title[0]" />
    <Textimg :imgURL="imgURL[1]" :title="title[1]" />
  </div>
</template>

<script>
import Textimg from "@/components/Text-img.vue";

export default {
  name: "About",
  data() {
    return {
      imgURL: ["fgfdg.jpg", "image-icon.svg"],
      title: ["Test1", "Test2"],
    };
  },

  components: {
    Textimg,
  },
};
</script>

<style>
.about {
  display: flex;
  justify-content: space-around;
}
</style>
