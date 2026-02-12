import { defineConfig } from 'vitepress'
import container from 'markdown-it-container'

export default defineConfig({
  title: '强化学习导论',
  description: 'Reinforcement Learning: An Introduction 中英双语版',
  lang: 'zh-CN',
  base: '/rl-book-bilingual/',

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
  ],

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '开始阅读', link: '/chapters/ch01-introduction' },
    ],
    sidebar: [
      {
        text: '前言',
        items: [
          { text: '第二版前言', link: '/chapters/preface-2nd-edition' },
          { text: '第一版前言', link: '/chapters/preface-1st-edition' },
          { text: '符号说明', link: '/chapters/notation' },
        ],
      },
      {
        text: 'I 表格型求解方法',
        items: [
          { text: '1 Introduction 引言', link: '/chapters/ch01-introduction' },
          { text: '2 Multi-armed Bandits 多臂赌博机', link: '/chapters/ch02-multi-armed-bandits' },
          { text: '3 Finite MDP 有限马尔可夫决策过程', link: '/chapters/ch03-finite-markov-decision-processes' },
          { text: '4 Dynamic Programming 动态规划', link: '/chapters/ch04-dynamic-programming' },
          { text: '5 Monte Carlo Methods 蒙特卡洛方法', link: '/chapters/ch05-monte-carlo-methods' },
          { text: '6 TD Learning 时序差分学习', link: '/chapters/ch06-temporal-difference-learning' },
          { text: '7 n-step Bootstrapping n步自举', link: '/chapters/ch07-n-step-bootstrapping' },
          { text: '8 Planning and Learning 规划与学习', link: '/chapters/ch08-planning-and-learning' },
        ],
      },
      {
        text: 'II 近似求解方法',
        items: [
          { text: '9 On-policy Prediction 同策略预测', link: '/chapters/ch09-on-policy-prediction-approximation' },
          { text: '10 On-policy Control 同策略控制', link: '/chapters/ch10-on-policy-control-approximation' },
          { text: '11 Off-policy Methods 离策略方法', link: '/chapters/ch11-off-policy-methods-approximation' },
          { text: '12 Eligibility Traces 资格迹', link: '/chapters/ch12-eligibility-traces' },
          { text: '13 Policy Gradient 策略梯度', link: '/chapters/ch13-policy-gradient-methods' },
        ],
      },
      {
        text: 'III 深入探讨',
        items: [
          { text: '14 Psychology 心理学', link: '/chapters/ch14-psychology' },
          { text: '15 Neuroscience 神经科学', link: '/chapters/ch15-neuroscience' },
          { text: '16 Applications 应用与案例', link: '/chapters/ch16-applications-case-studies' },
          { text: '17 Frontiers 前沿', link: '/chapters/ch17-frontiers' },
        ],
      },
      {
        text: '附录',
        items: [
          { text: '参考文献', link: '/chapters/references' },
        ],
      },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/' },
    ],
    outline: { level: [2, 3], label: '本页目录' },
    search: { provider: 'local' },
  },

  markdown: {
    math: true,
    config: (md) => {
      // Register ::: en, ::: zh, ::: notes custom containers
      for (const type of ['en', 'zh']) {
        md.use(container, type, {
          render(tokens: any[], idx: number) {
            if (tokens[idx].nesting === 1) {
              return `<div class="bilingual-${type}">\n`
            }
            return '</div>\n'
          },
        })
      }

      md.use(container, 'notes', {
        render(tokens: any[], idx: number) {
          if (tokens[idx].nesting === 1) {
            return '<div class="author-notes">\n'
          }
          return '</div>\n'
        },
      })
    },
  },
})
