<template>
  <div class="box">
    <h1>{{ title }}</h1>
    <div class="map" id="box" :style="boxStyle">
      <video :src="baseUrl + videoURL" class="video" controls>
        Your browser does not support the video tag.
      </video>
    </div>
  </div>
</template>

<script>
export default {
  props: ["videoURL", "title","shadowType"],
  computed: {
    boxStyle() {
      // В зависимости от типа тени возвращаем соответствующий стиль
      switch (this.shadowType) {
        case "light":
          return { boxShadow: "0px 50px 100px 10px rgba(245, 51, 51, 0.25)" };
        case "dark":
          return { boxShadow: "0px 50px 100px 10px rgba(67, 245, 51, 0.25)" };
        default:
          return {}; // Если тип тени не определен, возвращаем пустой стиль
      }
    },
  },
  data() {
    return {
      baseUrl: "http://26.234.143.237:8000",
    };
  },
};
</script>

<style scoped>
@import url("https://fonts.googleapis.com/css?family=Raleway:200");

.box {
  width: 40%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.img {
  border-radius: 10px;
}

.map {
  margin: 100px;
  width: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  --borderWidth: 10px;
  position: relative;
  border-radius: var(--borderWidth);
}

.map:after {
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

#box {
  color: #000000;
  font-family: "Raleway";
  font-size: 1rem;
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

.video {
  border-radius: calc(2 * var(--borderWidth));
}
</style>
