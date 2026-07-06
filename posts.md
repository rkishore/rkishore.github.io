# All Posts

A running index of everything on this blog, newest first.

{% for post in site.posts %}
- <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span> — [{{ post.title }}]({{ post.url | relative_url }})
{% endfor %}
